from relogic.logickit.training.trainer import Trainer

import torch
import json

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
  results = model_trainer.analyze_task(model_trainer.tasks[0], head_mask, head_importance, attn_entropy)
  return attn_entropy, head_importance, results


def mask_heads(config, model_trainer: Trainer):
  # _, head_importance, results = compute_heads_importance(config, model_trainer)

  bert_config = model_trainer.model.model.encoder.bert.config
  device = next(model_trainer.model.model.encoder.parameters()).device
  n_layers, n_heads = bert_config.num_hidden_layers, bert_config.num_attention_heads
  head_importance = torch.zeros(n_layers, n_heads).to(device)

  new_head_mask = torch.ones_like(head_importance)
  heads_to_mask = json.load(open(config.head_to_mask_file))
  for head_to_maks in heads_to_mask:
    new_head_mask[tuple(head_to_maks)] = 0.0
  print(new_head_mask)

  _, head_importance, results = compute_heads_importance(config, model_trainer, head_mask=new_head_mask)


