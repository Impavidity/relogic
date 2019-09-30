import torch.nn as nn

class SequenceLabelingModule(nn.Module):
  def __init__(self, config, task_name, n_classes):
    super(SequenceLabelingModule, self).__init__()
    self.config = config
    self.task_name = task_name
    self.n_classes = n_classes
    self.to_logits = nn.Linear(config.hidden_size, self.n_classes)

  def forward(self, *input, **kwargs):
    features = kwargs.pop("features")
    logits = self.to_logits(features)
    return logits