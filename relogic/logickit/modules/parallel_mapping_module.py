import torch.nn as nn
import torch
from relogic.logickit.base import utils
from relogic.logickit.utils import utils

class ParallelMappingModule(nn.Module):
  """

  """
  def __init__(self, config, task_name, n_classes=None):
    super(ParallelMappingModule, self).__init__()
    self.config = config
    self.task_name = task_name
    self.W = nn.Linear(in_features=config.hidden_size,
                       out_features=config.hidden_size,
                       bias=False)
    self.mode = self.config.parallel_mapping_mode

  def set_mode(self):
    if self.mode == "alignment":
      utils.log("Parallel Mapping Mode: Alignment")
      pass
    if self.mode == "adjustment":
      for param in self.W.parameters():
        param.requires_grad = False
      utils.log("Parallel Mapping Mode: Adjustment")

  def forward(self, *inputs, **kwargs):
    a_features = kwargs.pop("a_features")
    b_features = kwargs.pop("b_features")
    a_selected_indices = kwargs.pop("a_selected_indices")
    b_selected_indices = kwargs.pop("b_selected_indices")

    selected_indices_mask = (a_selected_indices > 0).float()

    a_selected_features = utils.batched_index_select(a_features[0], a_selected_indices)
    b_selected_features = utils.batched_index_select(b_features[0], b_selected_indices)

    if self.mode == "alignment":
      a = a_selected_features
      b = self.W(b_selected_features)
    elif self.mode == "adjustment":
      a = a_selected_features.detach()
      b = self.W(b_selected_features)
    else:
      raise ValueError("mode {} is not defined".format(self.mode))
    return (a-b) * selected_indices_mask.unsqueeze(-1)


