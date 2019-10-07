import torch.nn as nn
import torch.nn.functional as F
import torch
from relogic.logickit.base import utils
from relogic.logickit.utils import utils

class SelectIndexModule(nn.Module):
  def __init__(self, config, task_name, n_classes=None):
    super(SelectIndexModule, self).__init__()
    self.config = config
    self.task_name = task_name

  def forward(self, *inputs, **kwargs):
    student_results = kwargs.pop("student_results")
    teacher_results = kwargs.pop("teacher_results")
    # a_selected_indices = kwargs.pop("a_selected_indices")
    # b_selected_indices = kwargs.pop("b_selected_indices")

    # selected_indices_mask = (a_selected_indices > 0)

    # a_selected_features = utils.batched_index_select(teacher_results, a_selected_indices)
    # b_selected_features = utils.batched_index_select(student_results, b_selected_indices)
    a_selected_features = teacher_results[:, 0]
    b_selected_features = student_results[:, 0]

    # batch_size, sequence_length, dim -> batch_size, dim, sequence_length
    F.max_pool1d(teacher_results[:, 1:], )

    # return b_selected_features, a_selected_features, selected_indices_mask
    return b_selected_features, a_selected_features, None