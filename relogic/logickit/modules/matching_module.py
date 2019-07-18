import torch
import torch.nn as nn

class MatchingModule(nn.Module):
  def __init__(self, config, task_name, n_classes):
    super(MatchingModule, self).__init__()
    self.config = config
    self.task_name = task_name
    self.n_classes = n_classes
    self.to_logits = nn.Linear(config.hidden_size, self.n_classes)

  def forward(self, input, input_mask=None, segment_ids=None, extra_args=None):
    logits = self.to_logits(input[:,0])
    return logits