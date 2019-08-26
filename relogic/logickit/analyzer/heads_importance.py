from relogic.logickit.training.trainer import Trainer

import torch

def compute_heads_importance(config, model_trainer: Trainer, head_mask=None):
  """This method shows how to compute:
    - head attention entropy
    - head importance scores according to http://arxiv.org/abs/1905.10650
  """
  bert_config = model_trainer.model.model.encoder.bert.config
  device = next(model_trainer.model.model.encoder.parameters()).device
  n_layers, n_heads = bert_config.num_hidden_layers, bert_config.num_attention_heads
  head_importance = torch.zeros(n_layers, n_heads).to(device)
  attn_entropy = torch.zeros(n_layers, n_heads).to(device)

  if head_mask is None:
    head_mask = torch.ones(n_layers, n_heads).to(device)
  head_mask.requires_grad_(requires_grad=True)
  assert config.output_attentions, "You need to set output_attentions as True"
  model_trainer.analyze_task(model_trainer.tasks[0], head_mask, head_importance, attn_entropy)

