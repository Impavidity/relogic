from typing import Tuple, Union

import torch
import torch.nn as nn

from relogic.logickit.utils import utils

class Pruner(nn.Module):
  """
  This module scores and prunes items in a list using a parameterized scoring function
  and a threshold
  """
  def __init__(self, scorer: nn.Module) -> None:
    super(Pruner, self).__init__()
    self.scorer = scorer

  def forward(self,
              embeddings: torch.FloatTensor,
              mask: torch.LongTensor,
              num_items_to_keep: Union[int, torch.LongTensor]) -> Tuple[torch.FloatTensor,
      torch.LongTensor, torch.LongTensor, torch.FloatTensor, torch.FloatTensor]:
    """

    Args:
      embeddings: A tensor of shape (batch_size, num_items, embedding_size),
        containing an embedding for each item in the list that we want to prune.
      mask: A tensor of shape (batch_size, num_items), denoting unpadded elements
        of ``embeddings``.
      num_items_to_keep: If a tensor of shape (batch_size), specifies the number of
        items to keep for each individual sentence in minibatch.
        If an int, keep the same number of items for all sentences.

    """
    batch_size = mask.size(0)
    if isinstance(num_items_to_keep, int):
      num_items_to_keep = num_items_to_keep * torch.ones([batch_size],
                                                         dtype=torch.long, device=mask.device)
    max_items_to_keep = num_items_to_keep.max()

    # Shape (batch_size, num_items, 1)
    mask = mask.unsqueeze(-1)

    num_items = embeddings.size(1)
    # Shape (batch_size, num_items, 1)
    scores = self.scorer(embeddings)

    if scores.size(-1) != 1 or scores.dim() != 3:
      raise ValueError("The scorer passed to Pruner must produce a tensor of "
                       "shape ({}, {}, 1), but found shpae {}".format(batch_size, num_items, scores.size()))

    scores = utils.replace_masked_values(scores, mask, -1e20)
    # can not support k as tensor. So for each sentence, we keep the same size k which is max.
    _, top_indices = scores.topk(max_items_to_keep, 1)

    # Mask based on the number of items to keep for each sentence
    top_indices_mask = utils.get_mask_from_sequence_lengths(num_items_to_keep, max_items_to_keep)
    top_indices_mask = top_indices_mask.byte()

    # Shape (batch, max_items_to_keep)
    top_indices = top_indices.squeeze(-1)

    fill_value, _ = top_indices.max(dim=1)
    fill_value = fill_value.unsqueeze(-1)

    top_indices = torch.where(top_indices_mask, top_indices, fill_value)

    top_indices, _ = torch.sort(top_indices, 1)

    flat_top_indices = utils.flatten_and_batch_shift_indices(top_indices, num_items)

    top_embeddings = utils.batched_index_select(embeddings, top_indices, flat_top_indices)

    sequence_mask = utils.batched_index_select(mask, top_indices, flat_top_indices)
    sequence_mask = sequence_mask.squeeze(-1).byte()
    top_mask = top_indices_mask & sequence_mask
    top_mask = top_mask.long()

    # Shape: (batch_size, max_num_items_to_keep, 1)
    top_scores = utils.batched_index_select(scores, top_indices, flat_top_indices)

    return top_embeddings, top_mask, top_indices, top_scores, scores




