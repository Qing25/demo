import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "true"
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = FILE_DIR[:FILE_DIR.index('src')]

import re
import json
import random
from tqdm import tqdm

from transformers import BertTokenizerFast, BartTokenizer

import pytorch_lightning as pl
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from q_snippets.data import sequence_padding, BaseData, save_json, load_json
from q_snippets.object import Config, print_config, print_string

class TaskIdentifier:
    def __init__(self) -> None:
        self.taxons = [
            'VerifyStr', 'QueryRelationQualifier', 'VerifyDate', 'QueryAttrQualifier', 
            'Count', 'QueryAttr', 'What', 'QueryRelation', 
            'SelectAmong', 'VerifyNum', 'VerifyYear', 'SelectBetween'
        ]
        self.taxon2id = { n:i for i,n in enumerate(self.taxons)}

    def __call__(self, sample):
        func_name = sample['program'][-1]['function']
        return self.taxon2id[func_name]

    def _get_all_taxons(self):
        train = load_json("../../data/kqa/train.json")
        funcs = set([  sample['program'][-1]['function'] for sample in train  ])
        print(funcs)

@dataclass
class NL2SparqlSample:
    """{
            "id": "train_1",
            "program": [
                {
                    "function": "Find",
                    "dependencies": [],
                    "inputs": [
                        "Georgia national football team"
                    ]
                },
                {
                    "function": "QueryAttrQualifier",
                    "dependencies": [
                        0
                    ],
                    "inputs": [
                        "ranking",
                        "78",
                        "review score by"
                    ]
                }
            ],
            "sparql": "SELECT DISTINCT ?qpv WHERE { ?e <pred:name> \"Georgia national football team\" . ?e <ranking> ?pv . ?pv <pred:unit> \"1\" . ?pv <pred:value> \"78\"^^xsd:double . [ <pred:fact_h> ?e ; <pred:fact_r> <ranking> ; <pred:fact_t> ?pv ] <review_score_by> ?qpv .  }",
            "answer": "FIFA",
            "choices": [
                "Peter Travers",
                "Roger Ebert",
                "FIFA",
                "James Berardinelli",
                "Innovation, Science and Economic Development Canada",
                "Internet Movie Database",
                "Empire",
                "Charity Navigator",
                "Gene Siskel",
                "The World of Movies"
            ],
            "question": "Who is the reviewer of the Georgia national football team, which is ranked 78th?",
        },
    """
    id: str 
    question : str
    sparql : str 
    answer : str 
    category: int 


@dataclass
class NL2SparqlFeature:
    id: str 
    input_ids: list
    attention_mask : list 
    target_ids : list 
    category : int
    answer : str


@dataclass
class NL2SparqlBatch(BaseData):
    input_ids: torch.Tensor
    attention_mask:torch.Tensor
    target_ids:torch.Tensor
    answers : list

    def __len__(self):
        return self.input_ids.size(0)

    @classmethod
    def from_tensor_indice(cls, batch, indice):
        return cls(
            input_ids=batch.input_ids[indice],
            attention_mask=batch.attention_mask[indice],
            target_ids=batch.target_ids[indice],
            answers=batch.answers[indice]
        )
    
    @classmethod
    def chunk(cls, batch, n, dim):
        chunk_size = len(batch.answers) // n 
        chunks = {
            'input_ids': batch.input_ids.chunk(n, dim=dim), 'attention_mask':batch.attention_mask.chunk(n, dim=dim),
            'target_ids': batch.target_ids.chunk(n, dim=dim),
            'answers' : [batch.answers[i*chunk_size:(i+1)*chunk_size] for i in range(2)]
        }
        return (cls(**{ k:v[i] for k,v in chunks.items()} ) for i in range(n))

    def __getitem__(self, index):
        return NL2SparqlBatch(self.input_ids[index], self.attention_mask[index], self.target_ids[index], self.answers[index])

    def info(self):
        size = { k: v.size() if type(v) is torch.Tensor else len(v) for k,v in self.__dict__.items()}
        print(size)


class NL2SparqlDataset(Dataset):
    def __init__(self, config, tokenizer, dataset_mode) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.mode = dataset_mode
        self.task_identifier = TaskIdentifier()
        self.samples, self.features = self._handle_cache()

    def _handle_cache(self):
        """
            核心是 self.load_data 加载并处理数据，返回原始数据和处理后的特征数据
            需要注意缓存文件与 self.config.cache_dir  self.mode 有关
        Returns:
            samples, features
        """
        os.makedirs(self.config.cache_dir, exist_ok=True)               # 确保缓存文件夹存在
        file = os.path.join(self.config.cache_dir, f"{self.mode}.pt")   # 获得缓存文件的路径   
        if os.path.exists(file) and not self.config.force_reload:       # 如果已经存在，且没有强制重新生成，则从缓存读取
            samples, features = torch.load(file)
            print(f" {len(samples), len(features)} samples, features loaded from {file}")
            return samples, features
        else:
            samples, features = self.load_data()                        # 读取并处理数据
            torch.save((samples, features), file)                       # 生成缓存文件
            return samples, features


    def load_data(self):
        samples = []
        features = [] 
        if self.mode == 'train':
            path = self.config.train_path
        elif self.mode == 'val':
            path = self.config.val_path
        else:
            path = self.config.test_path
        with open(path, 'r', encoding='utf8') as f:
            obj = json.load(f)
            for sample in tqdm(obj):
            # for sample in tqdm(obj[:1000]):
                s = NL2SparqlSample(
                    id=sample['id'], question=sample['question'], sparql=sample.get('sparql', None), answer=sample.get('answer',None),
                    category=self.task_identifier(sample)
                )
                samples.append(s)
                q_td = self.tokenizer(s.question)
                feature = NL2SparqlFeature(
                    id=s.id, input_ids=q_td.input_ids, attention_mask=q_td.attention_mask, 
                    target_ids=self.tokenizer(s.sparql).input_ids if s.sparql is not None else None,
                    category=s.category, answer=s.answer
                )
                features.append(feature)
        return samples, features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]

    @staticmethod
    def collate_fn(batch):
        test = batch[0].target_ids is None
        max_input_len = max([len(feature.attention_mask) for feature in batch])
        max_target_len = max([len(feature.target_ids)  for feature in batch]) if not test else 0
        max_len = max(max_input_len, max_target_len)

        input_ids = torch.tensor(sequence_padding([x.input_ids for x in batch], length=max_len))
        attention_mask = torch.tensor(sequence_padding([x.attention_mask for x in batch], length=max_len))
        target_ids = torch.tensor(sequence_padding([x.target_ids for x in batch], length=max_len)) if not test else None
        batch = NL2SparqlBatch(
            input_ids=input_ids, attention_mask=attention_mask, target_ids=target_ids,
            answers=[ feature.answer for feature in batch]
        )
        return batch


class KqaNL2SparqlDataModule(pl.LightningDataModule):

    dataset_mapping = {
        'nl2sp' : NL2SparqlDataset
    }
    def __init__(self, config):
        super().__init__()
        self.config = config.data
        self.tokenizer = BartTokenizer.from_pretrained(config.data.tokenizer if config.data.tokenizer is not None else config.pretrained)
        print_string("configuration of datamodule")
        print_config(self.config)
        self.DATASET = self.dataset_mapping[self.config.dataset]
        
    def setup(self, stage=None):
        """
        根据模型运行的阶段，加载对应的数据，并实例化Dataset对象
        """
        if stage == 'fit' or stage is None:
            dataset = self.DATASET(self.config, self.tokenizer, dataset_mode='train')
            if self.config.val_path is None:
                print("number of samples in trainset : ", len(dataset))
                trainsize = int(0.8*len(dataset))
                trainset, valset = random_split(dataset, [trainsize, len(dataset)-trainsize])
                self.trainset, self.valset = trainset, valset 
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


def test_batch():
    
    config = Config.create(
        {
            'data': {
                'dataset': 'nl2sp',
                'tokenizer': "/pretrains/pt/bart_nl2sparql",
                'train_path': os.path.join(PROJ_DIR, "data/kqa/train.json"),
                'val_path': os.path.join(PROJ_DIR, "data/kqa/val.json"),
                'test_path': os.path.join(PROJ_DIR, "data/kqa/test.json"),
                'train_bsz': 4,
                'val_bsz': 4,
                'test_bsz': 4,
                'nways': 8,
                'kshots': 4,
                'cache_dir': './cached/data_utils',
                'force_reload': False
            },
        })
    dm = KqaNL2SparqlDataModule(config)
    dm.setup('fit')
    print_string("one sample example")
    print(dm.trainset.samples[0])
    print_string("one feature example")
    print(dm.trainset.features[0])
    print_string("one batch example")
    for batch in dm.train_dataloader():
        batch.info()
        print(batch)
        # print(len(batch), "len task_batch")
        # # split_batch(batch)
        # for task in batch:
        #     task.info()
        #     break
        break


if __name__ == '__main__':
    test_batch()