import torch
import torch.nn as nn

class ClassificationModule(nn.Module):
  def __init__(self, config, task_name, n_classes):
    super(ClassificationModule, self).__init__()
    self.config = config
    self.task_name = task_name
    self.n_classes = n_classes
    self.to_logits = nn.Linear(config.hidden_size, self.n_classes)

  def forward(self, *inputs, **kwargs):
    features = kwargs.pop("features")
    if isinstance(features, list):
      logits = self.to_logits(features[0][:, 0])
    else:
      logits = self.to_logits(features[:, 0])
    return logits