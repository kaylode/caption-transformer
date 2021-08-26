from .base_model import BaseModel
from models.transformer.search import sampling_search

import sys
sys.path.append('..')

class Captioning(BaseModel):
    def __init__(self, model, **kwargs):
        super(Captioning, self).__init__(**kwargs)
        self.model = model
        self.model_name = self.model.name
        if self.optimizer is not None:
            self.optimizer = self.optimizer(self.parameters(), lr= self.lr)
            self.set_optimizer_params()

        if self.freeze:
            for params in self.model.parameters():
                params.requires_grad = False

        if self.device:
            self.model.to(self.device)
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):

        src_inputs = batch['image_patches'].to(self.device)
        src_masks = batch['image_masks'].unsqueeze(-2).to(self.device)
        tgt_inputs = batch['texts_inp'].to(self.device)
        tgt_targets = batch['texts_res'].to(self.device)
        tgt_masks = batch['text_masks'].to(self.device)

        outputs = self.model(
            src = src_inputs, 
            trg = tgt_inputs, 
            src_mask = src_masks, 
            trg_mask = tgt_masks)

        loss = self.criterion(
                outputs.contiguous().view(-1, outputs.size(-1)), 
                tgt_targets.contiguous().view(-1))

        loss_dict = {'T': loss.item()}
        return loss, loss_dict

    def inference_step(self, batch, tgt_tokenizer):

        src_inputs = batch['image_patches'].to(self.device)
        src_masks = batch['image_masks'].unsqueeze(-2).to(self.device)

        outputs = self.model.predict(
            src_inputs = src_inputs,
            src_masks = src_masks, 
            tokenizer = tgt_tokenizer,
            max_len=64)

        return outputs  

    def evaluate_step(self, batch):
        src_inputs = batch['image_patches'].to(self.device)
        src_masks = batch['image_masks'].unsqueeze(-2).to(self.device)
        tgt_inputs = batch['texts_inp'].to(self.device)
        tgt_targets = batch['texts_res'].to(self.device)
        tgt_masks = batch['text_masks'].to(self.device)

        outputs = self.model(
            src = src_inputs, 
            trg = tgt_inputs, 
            src_mask = src_masks, 
            trg_mask = tgt_masks)

        loss = self.criterion(
                outputs.contiguous().view(-1, outputs.size(-1)), 
                tgt_targets.contiguous().view(-1))

        loss_dict = {'T': loss.item()}

        self.update_metrics(model=self)
        return loss, loss_dict

