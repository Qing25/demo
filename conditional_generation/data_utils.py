
from collections import defaultdict
from dataclasses import dataclass
import random
import os
import sys
# FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJ_DIR = FILE_DIR[:FILE_DIR.index('src')]
PROJ_DIR = os.path.abspath("..")
print(f"PROJ_DIR is: {PROJ_DIR}, adding to sys.path")
sys.path.append(PROJ_DIR)
from tqdm import tqdm

from transformers import BertTokenizerFast, BartTokenizer, T5Tokenizer
from transformers import BertTokenizer

import pytorch_lightning as pl
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from q_snippets.data import sequence_padding, BaseData, save_json, load_json
from q_snippets.object import Config, print_config, print_string


@dataclass
class Seq2seqSample:
    """

    """
    qid: str 
    input_seq: str 
    target_seq: str 


    @classmethod
    def from_dict(cls, d):
        obj = cls(d.get('qid'), d.get('input_seq' ), d.get('target_seq'))
        return obj


@dataclass
class Seq2seqFeature:
    input_ids : list 
    attention_mask : list 
    target_ids : list 


@dataclass
class Seq2seqBatch(BaseData):
    input_ids: torch.Tensor
    attention_mask:torch.Tensor
    target_ids:torch.Tensor

    def __len__(self):
        return self.input_ids.size(0)

    def info(self):
        info = { k: v.size() if type(v) is torch.Tensor else len(v) for k,v in self.__dict__.items()}
        print(info)


class DataReader:

    def _get_max_len(self, features):
        """ 
            KQA 预处理是找出问题和sparql都tokenize之后，最长的值，将所有的都pad到这个长度
        """
        max_len = 0
        for feature in features:
            q_len = len(feature.input_ids)
            t_len = len(feature.target_ids) if feature.target_ids is not None else 0
            _m = q_len if q_len > t_len else t_len 
            if _m > max_len:
                max_len = _m 
        return max_len

    def load_samples(self, path, mode):
        samples = []
        data = load_json(path)
        for i, sample in enumerate(data):
            
            _sample = Seq2seqSample.from_dict(sample)
            samples.append(_sample)
        return samples


    def load_features(self, samples, tokenizer):
        """ 

        """
        features = []
        for sample in samples:
             
            input_td = tokenizer(sample.input_seq)   
            feature = Seq2seqFeature(
                input_ids=input_td.input_ids, attention_mask=input_td.attention_mask, 
                target_ids=tokenizer(sample.target_seq).input_ids if sample.target_seq is not None else None,
            )
            features.append(feature)
        return features

class Seq2seqDataset(Dataset):
    def __init__(self, config, tokenizer, dataset_mode) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.mode = dataset_mode
        self.datareader = DataReader()
        self.samples, self.features = self._handle_cache()
        self.max_len = self.datareader._get_max_len(self.features)
        print_string(f"max len is {self.max_len}")
        

    def _handle_cache(self):
        """
            核心是 self.load_data 加载并处理数据，返回原始数据和处理后的特征数据
            需要注意缓存文件与 self.config.cache_dir  self.mode 有关
        Returns:
            samples, features
        """
        os.makedirs(self.config.cache_dir, exist_ok=True)               # 确保缓存文件夹存在
        file_path = getattr(self.config, f'{self.mode}_path').split("/")[-1]
        file = os.path.join(self.config.cache_dir, f"{self.mode}_{file_path}.pt")   # 获得缓存文件的路径   
        if os.path.exists(file) and not self.config.force_reload:       # 如果已经存在，且没有强制重新生成，则从缓存读取
            samples, features = torch.load(file)
            print(f" {len(samples), len(features)} samples, features loaded from {file}")
            return samples, features
        else:
            samples, features = self.load_data()                        # 读取并处理数据
            torch.save((samples, features), file)                       # 生成缓存文件
            return samples, features

    def load_data(self):

        samples, features = [], []
        if self.mode == 'train':
            path = self.config.train_path
        elif self.mode == 'val':
            path = self.config.val_path
        else:
            path = self.config.test_path
        samples = self.datareader.load_samples(path, mode=self.mode)
        features = self.datareader.load_features(samples, self.tokenizer)

        return samples, features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


    def collate_fn(self, batch):
        test = batch[0].target_ids is None

        input_ids = torch.tensor(sequence_padding([x.input_ids for x in batch], length=self.max_len,force=True))
        attention_mask = torch.tensor(sequence_padding([x.attention_mask for x in batch], length=self.max_len,force=True))
        target_ids = torch.tensor(sequence_padding([x.target_ids for x in batch], length=self.max_len,force=True)) if not test else None 

        batch = Seq2seqBatch(
            input_ids=input_ids, attention_mask=attention_mask, 
            target_ids=target_ids
        )
        return batch


class AnotherVersionSeq2seqDataset(Seq2seqDataset):

    def load_data(self):
        samples, features = [], []
        if self.mode == 'train':
            path = self.config.train_path
        elif self.mode == 'val':
            path = self.config.val_path
        else:
            path = self.config.test_path
        samples = self.datareader.load_samples_another(path, mode=self.mode)        # 修改数据读取函数
        features = self.datareader.load_features_another(samples, self.tokenizer)   # 修改数据处理函数

        return samples, features


def get_tokenizer_by_name(config):
    if config.model_name =='bart':
        tokenizer = BartTokenizer.from_pretrained(config.pretrained)
    elif config.model_name == 't5':
        tokenizer = T5Tokenizer.from_pretrained(config.pretrained)
    else:
        raise Exception(f"no tokenizer specified for model_name : {config.model_name}")
    return tokenizer


class Seq2seqDataModule(pl.LightningDataModule):

    dataset_mapping = {
        'seq2seq': Seq2seqDataset,
        'another': AnotherVersionSeq2seqDataset
    }
    def __init__(self, config):
        super().__init__()
        self.config = config.data

        self.tokenizer = get_tokenizer_by_name(config)
        print_string("configuration of datamodule")
        print_config(self.config)
        self.DATASET = self.dataset_mapping[self.config.dataset]

    def setup(self, stage=None):
        """
        根据模型运行的阶段，加载对应的数据，并实例化Dataset对象
        """
        if stage == 'fit' or stage is None:
            dataset = self.DATASET(self.config, self.tokenizer, dataset_mode='train')
            if self.config.val_path is None:                                                # 建议手动分数据并提供val_path
                print("number of samples in trainset : ", len(dataset))
                trainsize = int(0.8*len(dataset))
                trainset, valset = random_split(dataset, [trainsize, len(dataset)-trainsize])
                self.trainset, self.valset = trainset, valset 
                self.trainset.collate_fn = self.valset.collate_fn = dataset.collate_fn       # Subset 类没有此属性，手动设置之
                print(f"No val_path provided, split train to {trainsize}, {len(dataset)-trainsize}")
            else:
                self.trainset = dataset
                self.valset = self.DATASET(self.config, self.tokenizer, dataset_mode='val')    
        
        if stage == 'test' or stage is None:
            self.testset = self.DATASET(self.config, self.tokenizer, dataset_mode='test')
            
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.config.train_bsz,
            shuffle=True,
            collate_fn=self.trainset.collate_fn,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.config.val_bsz, 
            shuffle=False,
            collate_fn=self.valset.collate_fn, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.config.test_bsz, 
            shuffle=False,
            collate_fn=self.testset.collate_fn, num_workers=4)

    def predict_dataloader(self):
        return self.test_dataloader()

config = Config.create({
    'model_name':'bart',
    'pretrained' : "/pretrains/pt/facebook-bart-base",
    "data":{
        'dataset': 'seq2seq',
        'tokenizer': "/pretrains/pt/bart-base",
        # 'tokenizer': os.path.join(PROJ_DIR, "ckpt/bart-base"),
        'train_path': os.path.join(PROJ_DIR, "data/demo/train.json"),
        'val_path': os.path.join(PROJ_DIR, "data/demo/val.json"),
        # 'val_path' : None,
        'test_path': os.path.join(PROJ_DIR, "data.val.json"),
        'train_bsz': 4,
        'val_bsz': 4,
        'test_bsz': 4,
        'nways': 8,
        'kshots': 4,
        'cache_dir': './cached/data_utils',
        'force_reload': True
    }
})

def test_batch():
    """
        写完以上内容后，测试效果
    """
    dm = Seq2seqDataModule(config)
    dm.setup('fit')
    print_string("one sample example")
    print(dm.trainset.samples[0])
    print_string("one feature example")
    print(dm.trainset.features[0])
    print_string("one batch example")
    for batch in dm.train_dataloader():
        batch.info()
        print(batch)
        break
        # print(dm.tokenizer.batch_decode(batch.target_ids.cpu().numpy().tolist()))
        input_seq = dm.tokenizer.batch_decode(batch.input_ids.cpu().numpy().tolist())
        if any( "<unk>" in x for x in input_seq ):
            print(input_seq)
            input()
        if torch.any(batch.input_ids == 3):
            print(input_seq)
            input()

if __name__ == '__main__':
    test_batch()