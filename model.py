
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from transformers import BartConfig, BartForConditionalGeneration

class BartGeneration(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.bart_config = BartConfig.from_pretrained(config.pretrained)
        self.model = BartForConditionalGeneration(self.bart_config)
        
    def forward(self, batch):
        if batch.target_ids is not None:
            target_ids = batch.target_ids[:, :-1].contiguous()
            lm_labels = batch.target_ids[:, 1:].clone()
            lm_labels[batch.target_ids[:, 1:] == self.bart_config.pad_token_id] = -100
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


if __name__ == '__main__':
    pass