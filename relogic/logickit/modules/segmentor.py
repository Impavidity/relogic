import torch.nn as nn


class Segmentor(nn.Module):
  """

  """
  def __init__(self, config, n_classes):
    super(Segmentor, self).__init__()
    self.config = config
    self.to_logits = nn.Linear(config.hidden_size, n_classes)

  def forward(self,
              input,
              extra_args=None):
    # pooling/average to get the word representation

    # to logits

    logits = None

    return logits