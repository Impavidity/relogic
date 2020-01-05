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
    elif config.doc_ir_model == "cls":
      self.to_logits = nn.Linear(config.hidden_size, self.n_classes)
    elif config.doc_ir_model == "evidence":
      self.to_logits = nn.Linear(config.hidden_size, self.n_classes)
      self.to_sequence_logits = nn.Linear(config.hidden_size, 3)
    else:
      pass


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
      if self.config.pooling_method == "min":
        masked_selected_spans_features = (1.0 - selected_indices_mask.unsqueeze(-1).float()) * 10000.0
        pooled = (selected_spans_features + masked_selected_spans_features).min(dim=1)[0]
        # token_a_spans = token_a_spans[selected_a_indices]
      elif self.config.pooling_method == "max":
        masked_selected_spans_features = (1.0 - selected_indices_mask.unsqueeze(-1).float()) * -10000.0
        pooled = (selected_spans_features + masked_selected_spans_features).max(dim=1)[0]
      elif self.config.pooling_method == "mean":
        pooled = selected_spans_features.sum(1) / torch.sum(selected_indices_mask.float(), dim=-1).unsqueeze(-1)

      logits = self.to_logits(torch.cat([features[:, 0], pooled], dim=-1))
    elif self.config.doc_ir_model == "cls":
      logits = self.to_logits(features[:, 0])
    elif self.config.doc_ir_model == "evidence":
      sequence_features = kwargs.pop("encoding_results").pop("selected_non_final_layers_features")[0]
      logits = self.to_logits(features[:, 0])
      sequence_logits = self.to_sequence_logits(sequence_features)
      if self.training:
        return logits, sequence_logits
      else:
        return logits
    else:
      raise NotImplementedError("Unknown model {}".format(self.config.doc_ir_model))

    return logits