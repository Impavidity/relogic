import torch
import torch.nn as nn
from relogic.logickit.utils.utils import masked_softmax
import math

class DotProductAttention(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, query, encoder_outputs, encoder_output_mask):
    # query (batch_size, hidden_size)
    # encoder_outputs (batch_size, sequence_length, hidden_size)
    # encoder_output_mask (batch_size, sequence_length)
    att_weight = torch.bmm(encoder_outputs, query.unsqueeze(-1)).squeeze(-1)
    if encoder_output_mask is not None:
      att_weight = masked_softmax(att_weight, encoder_output_mask)
    else:
      att_weight = torch.softmax(att_weight, dim=-1)
    # (batch_size, sequence_length)
    return att_weight

class AddictiveAttention(nn.Module):
  def __init__(self, hidden_size):
    super().__init__()
    self.linear = nn.Linear(hidden_size * 2, hidden_size, bias=False)
    self.project = nn.Linear(hidden_size, 1, bias=False)

  def forward(self, query, encoder_outputs, encoder_output_mask):
    batch_size, seq_length, hidden_size = encoder_outputs.size()
    query = query.unsqueeze(-2).expand(batch_size, seq_length, hidden_size)
    intermediate = self.linear(torch.cat([query, encoder_outputs], dim=-1))
    att_weight = self.project(torch.tanh(intermediate)).squeeze(-1)
    if encoder_output_mask is not None:
      att_weight = masked_softmax(att_weight, encoder_output_mask)
    else:
      att_weight = torch.softmax(att_weight, dim=-1)
    # (batch_size, sequence_length)
    return att_weight

class PointerNet(nn.Module):
  def __init__(self, candidate_dim, query_dim):
    super().__init__()
    self.encoding_linear = nn.Linear(candidate_dim, query_dim, bias=False)


  def forward(self, query, candidates):
    # This module is for batched operation.
    # However, we start to write this module for SQL parsing.
    # At that task, we process the example one by one. So there is no batch dim.
    using_unsqueeze = False
    if len(query.size()) == 1 and len(candidates.size()) == 2:
      query = query.unsqueeze(0)
      candidates = candidates.unsqueeze(0)
      # if candidate_masks is not None:
      #   candidate_masks = candidate_masks.unsqueeze(0)
      using_unsqueeze = True

    candidates = self.encoding_linear(candidates)
    att_weight = torch.bmm(candidates, query.unsqueeze(-1)).squeeze(-1)
    if using_unsqueeze:
      return att_weight.squeeze(0)
    else:
      return att_weight


