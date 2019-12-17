import torch
import torch.nn as nn
from relogic.logickit.utils import utils
import torch.nn.functional as F

class IRMatchingModule(nn.Module):
  def __init__(self, config, task_name, n_classes):
    super().__init__()
    self.config = config
    self.task_name = task_name
    if self.config.regression:
      self.n_classes = 1
    else:
      self.n_classes = n_classes
    self.to_logits = nn.Sequential(
      nn.Linear(config.hidden_size * 2, config.hidden_size),
      nn.ReLU(),
      nn.Linear(config.hidden_size, self.n_classes)
    )

  def forward(self, *inputs, **kwargs):
    features = kwargs.pop("features")

    text_a_indices = kwargs.pop("text_a_indices")
    text_b_indices = kwargs.pop("text_b_indices")
    text_a_mask = (text_a_indices > 0).float()
    text_b_mask = (text_b_indices > 0).float()
    text_a_lengths = torch.sum(text_a_mask, dim=-1)
    text_b_lengths = torch.sum(text_b_mask, dim=-1)

    batch_size = features.size(0)
    query_token_size = text_a_indices.size(1)
    doc_token_size = text_b_indices.size(1)
    query_vector = utils.batched_index_select_tensor(features, text_a_indices)
    doc_vector = utils.batched_index_select_tensor(features, text_b_indices)

    exp_query_vector = query_vector.unsqueeze(2).repeat(1, 1, doc_token_size, 1)
    exp_doc_vector = doc_vector.unsqueeze(1).repeat(1, query_token_size, 1, 1)
    sim = F.cosine_similarity(exp_query_vector.view(-1, query_vector.size(-1)),
      exp_doc_vector.view(-1, doc_vector.size(-1))).view(batch_size, query_token_size, doc_token_size)

    scaled_doc_vector = sim.unsqueeze(-1) * exp_doc_vector
    exp_text_b_mask = text_b_mask.unsqueeze(1).repeat(1, query_token_size, 1).unsqueeze(-1)
    exp_text_b_lengths = text_b_lengths.unsqueeze(-1).repeat(1, query_token_size).unsqueeze(-1)
    per_query_token_based_doc_repr = torch.sum(scaled_doc_vector * exp_text_b_mask, dim=-2) / exp_text_b_lengths

    exp_text_a_mask = text_a_mask.unsqueeze(-1)
    exp_text_a_lengths = text_a_lengths.unsqueeze(-1)
    query_based_doc_repr = torch.sum(per_query_token_based_doc_repr * exp_text_a_mask, dim=-2) / exp_text_a_lengths

    query_vector_avg = torch.sum(query_vector * exp_text_a_mask , dim=-2) / exp_text_a_lengths

    feat = torch.cat([query_vector_avg, query_based_doc_repr], dim=-1)
    logits = self.to_logits(feat)
    return logits