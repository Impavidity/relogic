import torch.nn as nn
import torch
from relogic.logickit.base.utils import log

class SequenceLabelingModule(nn.Module):
  def __init__(self, config, task_name, n_classes):
    super(SequenceLabelingModule, self).__init__()
    self.config = config
    self.task_name = task_name
    self.n_classes = n_classes
    if hasattr(self.config, "sequence_labeling_use_cls") and self.config.sequence_labeling_use_cls:
      self.mul = 2
      log("Use cls in sequence labeling")
    else:
      self.mul = 1
    self.projection = nn.Linear(config.hidden_size * self.mul, config.projection_size)
    self.to_logits = nn.Linear(config.projection_size, self.n_classes)
    self.init_weight()

  def init_weight(self):
    self.to_logits.bias.data.zero_()
    self.to_logits.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

  def forward(self, *input, **kwargs):
    features = kwargs.pop("features")
    if self.mul == 2:
      features = torch.cat([features, features[:, 0].unsqueeze(1).repeat(1, features.size(1), 1)], dim=-1)
    logits = self.to_logits(self.projection(features))
    return logits