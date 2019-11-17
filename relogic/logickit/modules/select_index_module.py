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

    if self.config.select_index_method == "avg":
      # # a is the teacher
      # teacher_mask = (kwargs["a_input_ids"] > 0).float()
      # student_mask = (kwargs["input_ids"] > 0).float()
      # # 101 CLS 102 SEP
      # teacher_CLS_mask = (kwargs["a_input_ids"] == 101).float()
      # teacher_SEP_mask = (kwargs["a_input_ids"] == 102).float()
      # student_CLS_mask = (kwargs["input_ids"] == 101).float()
      # student_SEP_mask = (kwargs["input_ids"] == 102).float()
      # teacher_length = torch.sum(kwargs["a_input_ids"] > 0, dim=-1) - 2
      # student_length = torch.sum(kwargs["input_ids"] > 0, dim=-1) - 2
      # # Remove the CLS and SEP

      # teacher_sum = torch.sum(teacher_results * (teacher_mask - teacher_CLS_mask - teacher_SEP_mask).unsqueeze(-1), -2)
      # student_sum = torch.sum(student_results * (student_mask - student_CLS_mask - student_SEP_mask).unsqueeze(-1), -2)
      # teacher_avg = teacher_sum / teacher_length.unsqueeze(-1).float()
      # student_avg = student_sum / student_length.unsqueeze(-1).float()

      teacher_mask = (kwargs["a_is_head"] == 1).float()
      student_mask = (kwargs["b_is_head"] == 1).float()
      teacher_length = torch.sum(teacher_mask, dim=-1)
      student_length = torch.sum(student_mask, dim=-1)
      teacher_sum = torch.sum(teacher_results * teacher_mask.unsqueeze(-1), -2)
      student_sum = torch.sum(student_results * student_mask.unsqueeze(-1), -2)
      teacher_avg = teacher_sum / teacher_length.unsqueeze(-1).float()
      student_avg = student_sum / student_length.unsqueeze(-1).float()



      return student_avg, teacher_avg, None
    elif self.config.select_index_method == "cls":
      a_selected_features = teacher_results[:, 0]
      b_selected_features = student_results[:, 0]
      return b_selected_features, a_selected_features, None
    else:
      raise NotImplementedError("The selection method is not defined")
    # selected_indices_mask = (a_selected_indices > 0)

    # a_selected_features = utils.batched_index_select(teacher_results, a_selected_indices)
    # b_selected_features = utils.batched_index_select(student_results, b_selected_indices)

    # teacher_seq_length = teacher_results.size(1)
    # student_seq_length = student_results.size(1)
    # # batch_size, sequence_length, dim -> batch_size, dim, sequence_length
    # a_maxpooling_features = F.max_pool1d(teacher_results[:, 1:].transpose(1, 2), kernel_size=teacher_seq_length-1).squeeze(-1)
    # b_maxpooling_features = F.max_pool1d(student_results[:, 1:].transpose(1, 2), kernel_size=student_seq_length-1).squeeze(-1)



    # return b_selected_features, a_selected_features, selected_indices_mask

    # return b_maxpooling_features, a_maxpooling_features, None