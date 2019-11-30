import torch.nn as nn
import torch


class LR(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.model = nn.Linear(in_features=config.hidden_size, out_features=1)

  def forward(self, *inputs, **kwargs):
    feature = kwargs.pop("features")
    if feature.dim() == 3:
      feature = torch.mean(feature, dim=1)
    output = self.model(feature)
    return output

