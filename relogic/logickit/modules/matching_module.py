import torch
import torch.nn as nn
from relogic.logickit.utils import utils
from relogic.logickit.modules.span_extractors.average_span_extractor import AverageSpanExtractor

class MatchingModule(nn.Module):
  def __init__(self, config, task_name, n_classes):
    super(MatchingModule, self).__init__()
    self.config = config
    self.task_name = task_name
    if self.config.regression:
      self.n_classes = 1
    else:
      self.n_classes = n_classes
    if config.doc_ir_model == "keyword_selection":
      self.to_logits = nn.Linear(config.hidden_size * 2, self.n_classes)
      self.average_span_extractor = AverageSpanExtractor(input_dim=config.hidden_size)
    else:
      self.to_logits = nn.Linear(config.hidden_size, self.n_classes)


  # def forward(self, input, input_mask=None, segment_ids=None, extra_args=None, **kwargs):
  def forward(self, *inputs, **kwargs):
    features = kwargs.pop("features")

    if self.config.doc_ir_model == "keyword_selection":
      token_a_spans = kwargs.pop("token_a_spans")
      selected_a_indices = kwargs.pop("selected_a_indices")
      masks = kwargs.pop("input_mask")
      selected_indices_mask = selected_a_indices >= 0
      selected_a_indices[selected_a_indices < 0] = 0
      selected_spans = utils.batched_index_select_tensor(token_a_spans, selected_a_indices)
      selected_spans_features = self.average_span_extractor(
        sequence_tensor=features,
        span_indices=selected_spans,
        sequence_mask=masks,
        span_indices_mask=selected_indices_mask)
      # (batch_size, selected_span_size, dim)
      masked_selected_spans_features = (1.0 - selected_indices_mask.unsqueeze(-1).float()) * 10000.0
      min_pooled = (selected_spans_features + masked_selected_spans_features).min(dim=1)[0]
      # token_a_spans = token_a_spans[selected_a_indices]

      logits = self.to_logits(torch.cat([features[:, 0], min_pooled], dim=-1))
    else:
      logits = self.to_logits(features[:, 0])

    return logits