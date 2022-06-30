import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl 

from transformers import BartForConditionalGeneration, BartConfig
from transformers import MT5ForConditionalGeneration, MT5Config


class CustomedModel(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()

    def forward(self, ):


        return


def get_model_by_name(config):
    """ 根据 config.model_name 返回实例化后的模型 """

    if config.model_name == 'bart':
        # bart_config = BartConfig.from_pretrained(config.pretrained)
        # bart_config = prepare_config(config.pretrained)
        # print(config)
        # model =  BartForConditionalGeneration(bart_config)
        model = BartForConditionalGeneration.from_pretrained(config.pretrained)
        # model.config = bart_config
    elif config.model_name == 't5':
        model = MT5ForConditionalGeneration.from_pretrained(config.pretrained)
    elif config.model_name == 'custom':
        model = CustomedModel(config)
    else:
        raise Exception(f"{config.model} is not supported!")
    return model 