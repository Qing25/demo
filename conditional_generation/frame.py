import os
import sys
from turtle import forward
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJ_DIR = FILE_DIR[:FILE_DIR.index('src')]
# sys.path.append(PROJ_DIR)

PROJ_DIR = os.path.abspath("..")
print(f"proj_dir is: {PROJ_DIR}, adding to sys.path")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl 
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.functional import bleu_score

from transformers import get_linear_schedule_with_warmup

from q_snippets.data import load_json, save_json

from data_utils import Seq2seqDataModule
from model import get_model_by_name

class Seq2seqGeneration(pl.LightningModule):
    def __init__(self, config, model) :
        super().__init__()
        self.config = config 
        self.model = model 
        self.save_hyperparameters(ignore='model')  # ignore model to avoid assigning model to Omegaconf when load_from_ckpt

        self.val_pred_ids = []
        self.val_target_ids = [] 
        self.gold_corpus = [] 
        self.pred_corpus = []

    def forward(self, batch):
        def _custom_forward(batch):
            if batch.target_ids is not None:
                target_ids = batch.target_ids[:, :-1].contiguous()
                lm_labels = batch.target_ids[:, 1:].clone()
                lm_labels[batch.target_ids[:, 1:] == self.model.config.pad_token_id] = -100
            else:
                target_ids, lm_labels = None, None
            # print(batch.input_ids.size(), target_ids.size(), lm_labels.size())

            output = self.model(
                input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                decoder_input_ids=target_ids,
                labels=lm_labels,
                # output_attentions=True  # for copy mechanism
            )
            return output

        def _default_forward(batch):
            """ 训练时模型会自动从labels参数右移得到decoder_input_ids """
            return self.model(batch.input_ids, attention_mask=batch.attention_mask, labels=batch.target_ids)

        return _default_forward(batch)

    def training_step(self, batch, batch_idx):
        output = self(batch)
        self.log('train_loss', output.loss, prog_bar=True, sync_dist=True)
        return output.loss


    def validation_step(self, batch, batch_idx) :
        output = self(batch)
        self.val_pred_ids.extend(output.logits.argmax(-1).cpu().numpy().tolist())

        # save gold ids for bleu computing
        if self.gold_corpus == [] and batch.target_ids is not None:
            self.val_target_ids.extend(batch.target_ids.cpu().numpy().tolist())
        # self.log('val_loss', output.loss, prog_bar=True, sync_dist=True) 

    def _save_val_result(self):

        self.gold_corpus = ["None" for _ in self.pred_corpus ] if self.gold_corpus == [] else self.gold_corpus
        R = []
        for p, sample, g in zip(self.pred_corpus, self.trainer.datamodule.valset.samples, self.gold_corpus):
            R.append(dict(
                **sample.__dict__, 
                **{
                    'expected': g,
                    'generated':p}
            ))
        # logdir = trainer.logger.log_dir if hasattr(trainer.logger, 'log_dir') else trainer.logger.save_dir
        logdir = self.trainer.logger.log_dir 
        filename = os.path.join(logdir, f"val_epoch{self.current_epoch:02}.json")
        save_json(R, filename)
        

    def validation_epoch_end(self, outputs):
        tokenizer = self.trainer.datamodule.tokenizer
        self.pred_corpus = tokenizer.batch_decode(self.val_pred_ids, skip_special_tokens = True, clean_up_tokenization_spaces = True)

        if self.gold_corpus == [] and self.val_target_ids != [] :
            self.gold_corpus = tokenizer.batch_decode(self.val_target_ids, skip_special_tokens = True, clean_up_tokenization_spaces = True)

        print(len(self.pred_corpus), len(self.gold_corpus))
        bleu = bleu_score(self.pred_corpus, [ [_] for _ in self.gold_corpus])
        self.log('val_bleu', bleu, prog_bar=True, sync_dist=True) 

        self._save_val_result()
        self.val_pred_ids, self.val_target_ids =[], []

    def _get_grouped_params(self):
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
        optimizer = optim.AdamW(self._get_grouped_params(), lr=self.config.lr)
        # return optimizer
        total_steps = int(len(self.trainer.datamodule.train_dataloader()) // self.config.accumulate_grads ) * self.config.max_epochs # accumulate_grads
        warmup_step =  int(total_steps * self.config.warmup_rate)
        # lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=steps_per_epoch*self.config.max_epochs)        
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=total_steps)        

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step', 'frequency': 1, 'strict': True, 'monitor': None}]  


    def predict_step(self, batch , batch_idx) :
        batch_pred_ids = self.model.generate(
                input_ids=batch.input_ids, max_length=500, use_cache=True)
        return batch_pred_ids.cpu().numpy().tolist()  

    def on_predict_epoch_end(self, results) -> None:
        """ results = [[ batch_result ]] 
            batch_result = [[],[],...]
            聚合每个predict_step的结果，解码并保存到文件
        """
        all_pred_ids = sum(results[0], [])
        preds = self.trainer.datamodule.tokenizer.batch_decode(all_pred_ids, skip_special_tokens = True, clean_up_tokenization_spaces = True)
        R = []
        for sample, p in zip(self.trainer.datamodule.testset.samples, preds):
            R.append(dict(
                    **sample.__dict__, 
                    **{
                        'generated':p}
                ))
        save_json(R, self.config.preds.result_path)
        return preds


        

def train_model(config):

    _model = get_model_by_name(config)
    model = Seq2seqGeneration(config, _model)  # 创建Lightning框架

    dm = Seq2seqDataModule(config=config)
   
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
        filename="{epoch}_{train_loss:.4f}",   # 模型保存名称， epoch信息以及验证集分数
        monitor='train_loss',                                     
        mode='min',
        save_top_k=3,                                          
        verbose=True,
    )
    es = EarlyStopping('train_loss', patience=10, mode='min')
    trainer = pl.Trainer(
        accumulate_grad_batches=config.accumulate_grads,
        logger=logger,
        num_sanity_val_steps=0,
        limit_train_batches=64,  # 限制训练集数量，方便快速调试
        # limit_val_batches=64,  # 一般直接用全量测试数据吧, 验证函数可能会报错
        max_epochs=config.max_epochs,
        callbacks=[ckpt_callback, es],
        accelerator="gpu", 
        devices=1
    )
    # dm.setup(stage='fit')
    trainer.fit(model, dm)


def predict_ckpt(config):
    dm = Seq2seqDataModule(config)
    dm.setup(stage='test')
    _model = get_model_by_name(config)
    model = Seq2seqGeneration.load_from_checkpoint(config.preds.ckpt_path, config=config, model=_model)
    trainer = pl.Trainer(accelerator="gpu", devices=1)
    x = trainer.predict(model, dm)  # 预测结果已经在on_predict_epoch_end中保存了

    print(type(x))

