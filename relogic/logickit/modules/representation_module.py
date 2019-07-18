import torch.nn as nn


class RepresentationModule(nn.Module):
  def __init__(self, config, task_name, repr_size):
    super(RepresentationModule, self).__init__()
    self.config = config
    self.task_name = task_name
    self.repr_size = repr_size
    self.to_repr = nn.Linear(config.hidden_size, self.repr_size)


  def forward(self, input, input_mask=None, segment_ids=None, extra_args=None):
    logits = self.to_repr(input[:, 0])
    return logits