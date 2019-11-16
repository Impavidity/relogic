import torch.nn as nn
import torch
from relogic.logickit.utils.utils import batched_index_select_tensor

class MixSentAlignmentModule(nn.Module):
  def __init__(self, config, task_name, n_classes=None):
    super().__init__()
    self.config = config
    self.task_name = task_name

  def forward(self, *input, **kwargs):
    student_results = kwargs.pop("student_results")
    teacher_results = kwargs.pop("teacher_results")

    logits_a = teacher_results["a"]
    logits_b = teacher_results["b"]
    logits_c = student_results

    span_a_selected_index = kwargs.pop("span_a_selected_index")
    span_b_selected_index = kwargs.pop("span_b_selected_index")
    span_c_a_selected_index = kwargs.pop("span_c_a_selected_index")
    span_c_b_selected_index = kwargs.pop("span_c_b_selected_index")
    # The selected index should be all greater than 0

    selected_logits_a = batched_index_select_tensor(logits_a, span_a_selected_index)
    selected_logits_b = batched_index_select_tensor(logits_b, span_b_selected_index)
    selected_logits_c_a = batched_index_select_tensor(logits_c, span_c_a_selected_index)
    selected_logits_c_b = batched_index_select_tensor(logits_c, span_c_b_selected_index)

    combined_c_teacher = torch.cat([selected_logits_a, selected_logits_b], dim=1)
    combined_c_student = torch.cat([selected_logits_c_a, selected_logits_c_b], dim=1)

    return combined_c_student, combined_c_teacher, None