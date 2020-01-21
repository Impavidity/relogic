import torch
import torch.nn as nn

class MatchingModule(nn.Module):
  def __init__(self, config, task_name, n_classes):
    super(MatchingModule, self).__init__()
    self.config = config
    self.task_name = task_name
    if self.config.regression:
      self.n_classes = 1
    else:
      self.n_classes = n_classes
    if self.config.ir_siamese:
      hidden_size = config.hidden_size * 3
    else:
      hidden_size = config.hidden_size
    self.to_logits = nn.Linear(hidden_size, self.n_classes)

  # def forward(self, input, input_mask=None, segment_ids=None, extra_args=None, **kwargs):
  def forward(self, *inputs, **kwargs):
    if self.config.ir_siamese:
      a_features = kwargs.pop("a_features")
      b_features = kwargs.pop("b_features")
      logits = self.to_logits(torch.cat([a_features[:, 0], b_features[:, 0], torch.abs(a_features[:, 0]-b_features[:, 0])], dim=-1))
    else:
      features = kwargs.pop("features")
      logits = self.to_logits(features[:,0])
    return logits