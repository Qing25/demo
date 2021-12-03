import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import json
import collections
import numpy as np
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torchmetrics import Accuracy
from torchmetrics.functional import bleu_score
from q_snippets.object import Config
from q_snippets.data import load_json, save_json

from model import BartGeneration
from data_utils import KqaNL2SparqlDataModule

class TaskFrame(pl.LightningModule):
    model_dict = {
        'raw' : BartGeneration
    }
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = self.model_dict[config.model](config)
        self.save_hyperparameters(config)

        self.val_pred_ids = []
        self.val_target_ids = []

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_index):
        output = self(batch)
        self.log('train_loss', output.loss, prog_bar=True)
        return output.loss

    def validation_step(self, batch, batch_index):
        output = self(batch)
        self.val_pred_ids.extend(output.logits.argmax(-1).cpu().numpy().tolist())
        self.val_target_ids.extend(batch.target_ids.cpu().numpy().tolist())
        self.log('val_loss', output.loss, prog_bar=True)


    def validation_epoch_end(self, outputs) -> None:

        def calc_bleu(pred_ids, target_ids):
            gold_corpus = [ [sample[:sample.index(2)+1 if 2 in sample else None] ] for sample in pred_ids]
            pred_corpus = [ sample[:sample.index(2)+1 if 2 in sample else None ] for sample in  target_ids]
            return bleu_score(gold_corpus, pred_corpus)
        
        bleu = calc_bleu(self.val_pred_ids, self.val_target_ids)
        self.log('val_bleu', bleu, prog_bar=True)
        self.val_pred_ids = []
        self.val_target_ids = []

    def get_grouped_params(self):
        no_decay = ["bias", "LayerNorm.weight"]

        # Group parameters to those that will and will not have weight decay applied
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters

    def configure_optimizers(self):
        # optimizer = optim.AdamW(self.parameters(), lr=self.config.lr)
        optimizer = optim.AdamW(self.get_grouped_params(), lr=self.config.lr)
        return optimizer

def train_model(config):
    model = TaskFrame(config)
    dm = KqaNL2SparqlDataModule(config=config)
    logger = TensorBoardLogger(
        save_dir="./lightning_logs/",
        name=None,                # 指定experiment, ./lightning_logs/exp_name/version_name
        version=config.version,   # 指定version, ./lightning_logs/version_name
    )

    # 设置保存模型的路径及参数
    CUR_DIR = os.getcwd()
    dirname = os.path.join(CUR_DIR, "./lightning_logs/", config.version)
    ckpt_callback = ModelCheckpoint(
        dirpath=dirname,
        filename="{epoch}_{val_bleu:.3f}",   # 模型保存名称， epoch信息以及验证集分数
        monitor='val_bleu',                                     
        mode='max',
        save_top_k=3,                                          
        verbose=True,
    )
    
    # 设置训练器
    lrm = LearningRateMonitor('step')
    es = EarlyStopping('val_bleu', patience=30)
    trainer = pl.Trainer(
        accumulate_grad_batches=config.accumulate_grads,
        logger=logger,
        num_sanity_val_steps=0,
        limit_train_batches=64,  # 限制训练集数量，方便快速调试
        # limit_val_batches=64,  # 一般直接用全量测试数据吧, 验证函数可能会报错
        max_epochs=config.max_epochs,
        callbacks=[ckpt_callback, es, lrm],
        gpus=1,
        deterministic=True,
        # reload_dataloaders_every_epoch=True
    )
    # 开始训练模型
    dm.setup('fit')
    trainer.fit(model, dm)


class Predictor():
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        yaml_config = os.path.join(os.path.dirname(config.preds.ckpt_path), "hparams.yaml")
        if os.path.exists(yaml_config):
            _config = Config.load(yaml_config)
            _config.preds.update(config.preds)
            _config.mode = 'predict'
            print(f"config updated:\n{_config}", end="\n"+"="*20+"\n")
        self.config = _config
        self.model = TaskFrame.load_from_checkpoint(config.preds.ckpt_path, config=self.config).to(self.device)
        self.dm = KqaNL2SparqlDataModule(config=config)
        self.dm.setup('test')
    
    def predict(self):
        all_outputs = []
        all_answers = [ sample.answer for sample in self.predset.samples]
        with torch.no_grad():
            # for batch in tqdm(pred_loader, total=10):
            for batch in tqdm(self.dm.test_dataloader(), total=len(self.dm.test_dataloader())):
                # batch = batch.to(self.device)
                input_ids = batch.input_ids.to(self.device)
                outputs = self.model.model.generate(input_ids=input_ids, max_length=500, use_cache=True)
                # outputs = self.model.model.generate(
                #     input_ids=input_ids, max_length=500, use_cache=True,
                #     decoder_use_copy=True, output_attentions=True, output_hidden_states=True)
                all_outputs.extend(outputs.cpu().numpy())
            outputs = [self.tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for output_id in all_outputs]
            # pred_cyphers = [CypherUtil.post_process_pred_cypher(output) for output in outputs]
            res = []
            for p, a in zip(outputs, self.predset.samples):
                res.append({
                    'pred_cypher': p,  "pred_sp": "",
                    'sp' : a.sparql,
                    'answer': a.answer,
                    'question': a.question
                })
            save_json(res, "prediction_kqa_nl2cy_rawbart.json")

def run_prediction(config):
    p = Predictor(config)
    p.predict()