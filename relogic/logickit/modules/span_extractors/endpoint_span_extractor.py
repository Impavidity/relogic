import torch.nn as nn
import torch
from relogic.logickit.utils import utils


class EndpointSpanExtractor(nn.Module):
  def __init__(self,
               input_dim: int,
               combination: str = "x,y",
               num_width_embeddings: int = None,
               span_width_embedding_dim: int = None,
               bucket_widths: bool = False) -> None:
    super(EndpointSpanExtractor, self).__init__()
    self._input_dim = input_dim
    self._combination = combination
    self._num_width_embeddings = num_width_embeddings
    self._bucket_widths = bucket_widths

    if num_width_embeddings is not None and span_width_embedding_dim is not None:
      self._span_width_embedding = nn.Embedding(num_width_embeddings, span_width_embedding_dim)
    elif not all([num_width_embeddings is None, span_width_embedding_dim is None]):
      raise ValueError("To use a span width embedding representation, you must"
                               "specify both num_width_embeddings and span_width_embedding_dim.")
    else:
      self._span_width_embedding = None


  def forward(self,
              sequence_tensor: torch.FloatTensor,
              span_indices: torch.LongTensor,
              sequence_mask: torch.LongTensor = None,
              span_indices_mask: torch.LongTensor = None):
    span_starts, span_ends = [index.squeeze(-1) for index in span_indices.split(1, dim=-1)]

    if span_indices_mask is not None:
      span_starts = span_starts * span_indices_mask.long()
      span_ends = span_ends * span_indices_mask.long()

    # The span is exclusive on the right, so the span_ends need to -1

    start_embeddings = utils.batched_index_select(sequence_tensor, span_starts)
    inclusive_span_ends = torch.relu((span_ends - 1).float()).long()
    end_embeddings = utils.batched_index_select(sequence_tensor ,inclusive_span_ends)

    combined_tensors = torch.cat([start_embeddings, end_embeddings], dim=-1)

    if self._span_width_embedding is not None:
      # Embed the span widths and concatenate to the rest of the representations.
      if self._bucket_widths:
        span_widths = utils.bucket_values(span_ends - span_starts,
                                          num_total_buckets=self._num_width_embeddings)
      else:
        span_widths = span_ends - span_starts

      span_width_embeddings = self._span_width_embedding(span_widths)
      combined_tensors = torch.cat([combined_tensors, span_width_embeddings], dim=-1)

    if span_indices_mask is not None:
      return combined_tensors * span_indices_mask.unsqueeze(-1).float()

    return combined_tensors




