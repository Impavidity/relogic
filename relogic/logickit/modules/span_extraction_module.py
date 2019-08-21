import torch
import torch.nn as nn


class SpanExtractionModule(nn.Module):
  def __init__(self, config, task_name, n_classes):
    super(SpanExtractionModule, self).__init__()
    self.config = config
    self.task_name = task_name
    self.n_classes = n_classes
    self.to_logits = nn.Linear(config.hidden_size, self.n_classes)
    self.apply(self.init_weights)

  def init_weights(self, module):
    if isinstance(module, nn.Linear):
      # Slightly different from the TF version which uses truncated_normal for initialization
      # cf https://github.com/pytorch/pytorch/pull/5617
      module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    if isinstance(module, nn.Linear) and module.bias is not None:
      module.bias.data.zero_()


  def forward(self, input_feature, input_mask=None, segment_ids=None, extra_args=None, **kwargs):
    logits = self.to_logits(input_feature)
    # (batch_size, sentence_length, 2)
    start_logits, end_logits = logits.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1)
    end_logits = end_logits.squeeze(-1)

    return start_logits, end_logits


