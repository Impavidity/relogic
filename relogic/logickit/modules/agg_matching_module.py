import torch
import torch.nn as nn

class AggMatchingModule(nn.Module):
  def __init__(self, config, task_name, n_classes):
    super().__init__()
    self.config = config
    self.task_name = task_name
    self.n_classes = n_classes
    self.to_logits = nn.Linear(config.hidden_size, self.n_classes)

  # def forward(self, input, input_mask=None, segment_ids=None, extra_args=None, **kwargs):
  def forward(self, *inputs, **kwargs):
    features = kwargs.pop("features")
    features = features[:, 0]
    mean_feature = torch.mean(features, dim=0, keepdim=True)
    logits = self.to_logits(mean_feature)
    return logits

