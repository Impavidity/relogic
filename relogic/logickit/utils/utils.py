import numpy as np
import torch
from typing import Optional

def softmax(x):
  """Compute softmax values for each sets of scores in x."""
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=-1)

def gen_position_indicator(span, length):
  indicator = [0] * length
  for idx, i in enumerate(range(span[0], -1, -1)):
    indicator[i] = -idx
  for idx, i in enumerate(range(span[1], length)):
    indicator[i] = idx + 1
  return indicator

def indicator_vector(index, length, default_label=0, indicator_label=1, head_index=None):
  vector = [default_label] * length
  if head_index is None:
    for idx in index:
      vector[idx] = indicator_label
  else:
    for idx in index:
      vector[head_index[idx]] = indicator_label
  return vector

def truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

def get_span_labels(sentence_tags, is_head=None, segment_id=None, inv_label_mapping=None, ignore_label=list(["V"])):
  """Go from token-level labels to list of entities (start, end, class)."""
  if inv_label_mapping:
    sentence_tags = [inv_label_mapping[i] for i in sentence_tags]
  filtered_sentence_tag = []
  if is_head:
    # assert(len(sentence_tags) == len(is_head))

    for idx, (head, segment) in enumerate(zip(is_head, segment_id)):
      if head == 1 and segment == 0:
        if sentence_tags[idx] != 'X':
          filtered_sentence_tag.append(sentence_tags[idx])
        else:
          filtered_sentence_tag.append("O")
  if filtered_sentence_tag:
    sentence_tags = filtered_sentence_tag
  span_labels = []
  last = 'O'
  start = -1
  for i, tag in enumerate(sentence_tags):
    items = (None, 'O') if tag == 'O' else tag.split('-', 1)
    pos, _ = items if len(items) == 2 else (items[0], None)
    if (pos == 'S' or pos == 'B' or tag == 'O') and last != 'O':
      span_labels.append((start, i - 1, None if len(last.split('-', 1)) != 2 else last.split('-', 1)[-1]))
    if pos == 'B' or pos == 'S' or last == 'O':
      start = i
    last = tag
  if sentence_tags[-1] != 'O':
    span_labels.append((start, len(sentence_tags) - 1,
                        None if len(last.split('-', 1)) != 2 else last.split('-', 1)[-1]))
  for item in span_labels:
    if item[2] in ignore_label:
      span_labels.remove(item)
  return set(span_labels), sentence_tags

def filter_head_prediction(sentence_tags, is_head):
  filtered_sentence_tag = []
  for idx, head in enumerate(is_head):
    if head == 1:
      if sentence_tags[idx] != 'X':
        filtered_sentence_tag.append(sentence_tags[idx])
      else:
        filtered_sentence_tag.append("O")
  return filtered_sentence_tag

def create_tensor(features, attribute, dtype, device):
  return torch.tensor([getattr(f, attribute) for f in features], dtype=dtype).to(device)

def get_range_vector(size: int, device) -> torch.Tensor:
  """
  """
  return torch.arange(0, size, dtype=torch.long).to(device)

def flatten_and_batch_shift_indices(indices: torch.LongTensor,
                                    sequence_length: int) -> torch.Tensor:
  """``indices`` of size ``(batch_size, d_1, ..., d_n)`` indexes into dimension 2 of a target tensor,
  which has size ``(batch_size, sequence_length, embedding_size)``. This function returns a vector
  that correctly indexes into the flattened target. The sequence length of the target must be provided
  to compute the appropriate offset.

  Args:
    indices (torch.LongTensor):

  """
  if torch.max(indices) >= sequence_length or torch.min(indices) < 0:
    raise ValueError("All the elements should be in range (0, {}), but found ({}, {})".format(
      sequence_length - 1, torch.min(indices).item(), torch.max(indices).item()))
  offsets = get_range_vector(indices.size(0), indices.device) * sequence_length
  for _ in range(len(indices.size()) - 1):
    offsets = offsets.unsqueeze(1)

  # (batch_size, d_1, ..., d_n) + (batch_size, 1, ..., 1)
  offset_indices = indices + offsets

  # (batch_size * d_1 * ... * d_n)
  offset_indices = offset_indices.view(-1)
  return offset_indices


def batched_index_select(target: torch.Tensor,
                         indices: torch.LongTensor,
                         flattened_indices: Optional[torch.LongTensor] = None) -> torch.Tensor:
  """Select ``target`` of size ``(batch_size, sequence_length, embedding_size)`` with ``indices`` of
  size ``(batch_size, d_1, ***, d_n)``.

  Args:
    target (torch.Tensor): A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).

  """
  if flattened_indices is None:
    flattened_indices = flatten_and_batch_shift_indices(indices, target.size(1))

  # Shape: (batch_size * sequence_length, embedding_size)
  flattened_target = target.view(-1, target.size(-1))

  # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
  flattened_selected = flattened_target.index_select(0, flattened_indices)
  selected_shape = list(indices.size()) + [target.size(-1)]

  # Shape: (batch_size, d_1, ..., d_n, embedding_size)
  selected_targets = flattened_selected.view(*selected_shape)
  return selected_targets

def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
  """
  ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
  masked. This performs a softmax on just the non-masked positions of ``vector``. Passing ``None``
  in for the mask is also acceptable, which is just the regular softmax.

  """
  if mask is None:
    result = torch.softmax(vector, dim=dim)
  else:
    mask = mask.float()
    while mask.dim() < vector.dim():
      mask = mask.unsqueeze(1)
    masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
    result = torch.softmax(masked_vector, dim=dim)
  return result

def weighted_sum(matrix: torch.Tensor,
                 attention: torch.Tensor) -> torch.Tensor:
  """

  Args:
    matrix ():
    attention ():

  """
  if attention.dim() == 2 and matrix.dim() == 3:
    return attention.unsqueeze(1).bmm(matrix).squeeze(1)
  if attention.dim() == 3 and matrix.dim() == 3:
    return attention.bmm(matrix)
  if matrix.dim() - 1 < attention.dim():
    expanded_size = list(matrix.size())
    for i in range(attention.dim() - matrix.dim() + 1):
      matrix = matrix.unsqueeze(1)
      expanded_size.insert(i + 1, attention.size(i + 1))
    matrix = matrix.expand(*expanded_size)
  intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
  return intermediate.sum(dim=-2)


def replace_masked_values(tensor: torch.Tensor, mask: torch.Tensor, replace_with: float) -> torch.Tensor:
  """
  """
  if tensor.dim() != mask.dim():
    raise ValueError("tensor.dim() {} != mask.dim() {}.".format(tensor.dim(), mask.dim()))
  return tensor.masked_fill((1-mask).byte(), replace_with)

def get_mask_from_sequence_lengths(sequence_lengths: torch.Tensor, max_length: int) -> torch.Tensor:
  """Generate mask from variable ``(batch_size,)`` which represents the sequence lengths of each
  batch element.

  Returns:
    torch.Tensor: ``(batch_size, max_length)``
  """
  ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
  range_tensor = ones.cumsum(dim=1)
  return (range_tensor <= sequence_lengths.unsqueeze(1)).long()