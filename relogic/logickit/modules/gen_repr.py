import torch.nn as nn
import torch.nn.functional as F
import torch
from relogic.logickit.base import utils
from relogic.logickit.utils import utils

class GenRepr(nn.Module):
  def __init__(self, config, task_name, n_classes=None):
    super(GenRepr, self).__init__()
    self.config = config
    self.task_name = task_name

  def forward(self, *inputs, **kwargs):
    features = kwargs.pop("features")

    text_mask = (kwargs["input_mask"]).float()

    features_sum = torch.sum(features * text_mask.unsqueeze(-1), -2)
    length = torch.sum(kwargs["input_mask"] > 0, dim=-1)
    features_avg = features_sum / length.unsqueeze(-1).float()

    return features_avg