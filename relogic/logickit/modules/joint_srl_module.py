import torch.nn as nn
import torch
from relogic.logickit.modules.span_extractors.endpoint_span_extractor import EndpointSpanExtractor
from relogic.logickit.modules.span_extractors.self_attentive_span_extractor import SelfAttentiveSpanExtractor
from relogic.logickit.modules.span_extractors.average_span_extractor import AverageSpanExtractor
from relogic.logickit.modules.pruner import Pruner
from relogic.logickit.inference.modeling import gelu
from relogic.logickit.utils import utils
from relogic.logickit.base.constants import SPAN_REPR_KENTON_LEE, SPAN_REPR_AVE_MAX, SPAN_REPR_AVE

class JointSRLModule(nn.Module):

  def __init__(self, config, task_name, n_classes):
    super(JointSRLModule, self).__init__()
    self.config = config
    self.task_name = task_name
    self.n_classes = n_classes
    self.endpoint_span_extractor = EndpointSpanExtractor(
      input_dim=config.hidden_size,
      num_width_embeddings=config.num_width_embeddings,
      span_width_embedding_dim=config.span_width_embedding_dim)
    self.endpoint_predicate_span_extractor = EndpointSpanExtractor(
      input_dim=config.hidden_size)
    self.attentive_span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
    # self.average_span_extractor = AverageSpanExtractor(input_dim=config.hidden_size)
    if self.config.srl_arg_span_repr == SPAN_REPR_AVE:
      arg_span_dim = config.hidden_size
    elif self.config.srl_arg_span_repr == SPAN_REPR_AVE_MAX:
      arg_span_dim = config.hidden_size * 2
    elif self.config.srl_arg_span_repr == SPAN_REPR_KENTON_LEE:
      arg_span_dim = config.hidden_size * 3 + config.span_width_embedding_dim
    else:
      raise ValueError("The span repr is not defined {}".format(self.config.srl_arg_span_repr))

    if self.config.srl_pred_span_repr == SPAN_REPR_AVE:
      pred_span_dim = config.hidden_size
    elif self.config.srl_pred_span_repr == SPAN_REPR_AVE_MAX:
      pred_span_dim = config.hidden_size * 2
    elif self.config.srl_pred_span_repr == SPAN_REPR_KENTON_LEE:
      pred_span_dim = config.hidden_size * 2
    else:
      raise ValueError("The span repr is not defined {}".format(self.config.srl_pred_span_repr))

    arg_feedforward_scorer = nn.Sequential(
      nn.Linear(in_features=arg_span_dim,
                out_features=config.hidden_size),
      nn.ReLU(),
      nn.Linear(in_features=config.hidden_size, out_features=1))
    self.mention_pruner = Pruner(arg_feedforward_scorer)
    predicate_feedforward_scorer = nn.Sequential(
      nn.Linear(in_features=pred_span_dim, out_features=config.hidden_size),
      nn.ReLU(),
      nn.Linear(in_features=config.hidden_size, out_features=1))
    self.predicate_pruner = Pruner(predicate_feedforward_scorer)
    self.semantic_role_predictor = nn.Sequential(
      nn.Linear(in_features=arg_span_dim + pred_span_dim,
                out_features=config.hidden_size),
      nn.ReLU(),
      nn.Linear(in_features=config.hidden_size, out_features=self.n_classes-1))
    self.constant_pred = nn.Parameter(torch.LongTensor([15]), requires_grad=False)
    self.constant_arg = nn.Parameter(torch.LongTensor([30]), requires_grad=False)

  def forward(self, *inputs, **kwargs):
    features = kwargs.pop("features")
    arg_candidates = kwargs.pop("arg_candidates")
    predicate_candidates = kwargs.pop("predicate_candidates")

    num_arg_spans = arg_candidates.size(1)
    num_preds_spans = predicate_candidates.size(1)

    # Span padding is (1, 0). So we use [:, :, 1] > 0 can determine
    # the the span is padding or not.
    arg_mask = (arg_candidates[:, :, 1] > 0).float()
    predicate_mask = (predicate_candidates[:, :, 1] > 0).float()

    if self.config.srl_arg_span_repr == SPAN_REPR_KENTON_LEE:
      endpoint_span_embeddings = self.endpoint_span_extractor(
        sequence_tensor=features, span_indices=arg_candidates, span_indices_mask=arg_mask)

      attended_span_embeddings = self.attentive_span_extractor(
        sequence_tensor=features, span_indices=arg_candidates, span_indices_mask=arg_mask)

      arg_embeddings = torch.cat([endpoint_span_embeddings, attended_span_embeddings], -1)
    elif self.config.srl_arg_span_repr == SPAN_REPR_AVE_MAX:
      average_span_embeddings = self.average_span_extractor(
        sequence_tensor=features, span_indices=arg_candidates, span_indices_mask=arg_mask
      )
      max_pooled_span_embedding = self.max_pooling_extractor(

      )
      arg_embeddings = torch.cat([average_span_embeddings, max_pooled_span_embedding], -1)
    elif self.config.srl_arg_span_repr == SPAN_REPR_AVE:
      arg_embeddings = self.average_span_extractor(
        sequence_tensor=features, span_indices=arg_candidates, span_indices_mask=arg_mask
      )
    else:
      raise ValueError("The span repr is not defined {}".format(self.config.srl_arg_span_repr))

    num_spans_to_keep = (torch.sum(arg_mask, dim=-1) * 0.8).long()
    num_spans_to_keep = torch.min(num_spans_to_keep, self.constant_arg) # quick fix

    (top_arg_span_embeddings, top_arg_span_mask,
     top_arg_span_indices, top_arg_span_mention_scores,
     arg_span_mention_full_scores) = self.mention_pruner(
      arg_embeddings, arg_mask, num_spans_to_keep)
    # top_arg_span_mask = (batch_size, max_arg_to_keep)

    if self.config.srl_pred_span_repr == SPAN_REPR_KENTON_LEE:
      pred_embeddings = self.endpoint_predicate_span_extractor(
        sequence_tensor=features, span_indices=predicate_candidates, span_indices_mask=predicate_mask)
    elif self.config.srl_pred_span_repr == SPAN_REPR_AVE_MAX:
      average_pred_embeddings = self.average_sapn_extractor(
        sequence_tensor=features, span_indices=predicate_candidates, span_indices_mask=predicate_mask)
      max_pooled_pred_embeddings = self.max_pooling_extractor(

      )
      pred_embeddings = torch.cat([average_pred_embeddings, max_pooled_pred_embeddings], -1)
    elif self.config.srl_pred_span_repr == SPAN_REPR_AVE:
      pred_embeddings = self.average_span_extractor(
        sequence_tensor=features, span_indices=predicate_candidates, span_indices_mask=predicate_mask)
    else:
      raise ValueError("The span repr is not defined {}".format(self.config.srl_pred_span_repr))


    num_preds_to_keep = (torch.sum(predicate_mask, dim=-1) * 0.4).long()
    num_preds_to_keep = torch.min(num_preds_to_keep, self.constant_pred)

    (top_pred_span_embeddings, top_pred_span_mask,
     top_pred_span_indices, top_pred_span_mention_scores,
     pred_span_mention_full_scores) = self.predicate_pruner(
      pred_embeddings, predicate_mask, num_preds_to_keep)

    # top_arg_span_mask = top_arg_span_mask.unsqueeze(-1)
    flat_top_arg_span_indices = utils.flatten_and_batch_shift_indices(top_arg_span_indices, num_arg_spans)
    # Shape: (batch, max_arg_to_keep, 2)
    top_arg_spans = utils.batched_index_select(arg_candidates,
                                              top_arg_span_indices,
                                              flat_top_arg_span_indices)

    flat_top_pred_span_indices = utils.flatten_and_batch_shift_indices(
      top_pred_span_indices, num_preds_spans)
    # Shape: (batch, max_pred_to_keep, 2)
    top_pred_spans = utils.batched_index_select(predicate_candidates,
                                                top_pred_span_indices,
                                                flat_top_pred_span_indices)

    # Shape: (batch_size, num_pred_to_keep, num_arg_to_keep, embedding_size)
    span_pair_embeddings = self.compute_span_pair_embeddings(top_pred_span_embeddings,
                                                             top_arg_span_embeddings)

    srl_scores = self.compute_srl_scores(span_pair_embeddings,
                                         top_pred_span_mention_scores,
                                         top_arg_span_mention_scores)
    return (srl_scores, top_pred_spans,
            top_arg_spans, top_pred_span_mask, top_arg_span_mask,
            pred_span_mention_full_scores, arg_span_mention_full_scores)

  def compute_span_pair_embeddings(self,
                                   top_pred_span_embeddings: torch.FloatTensor,
                                   top_arg_span_embeddings: torch.FloatTensor) -> torch.FloatTensor:
    # Shape (batch_size, num_pred_to_keep, 1, embedding_size)
    num_pred_to_keep = top_pred_span_embeddings.size(1)
    num_arg_to_keep = top_arg_span_embeddings.size(1)
    expanded_pred_embeddings = top_pred_span_embeddings.unsqueeze(2).repeat(1, 1, num_arg_to_keep, 1)
    expanded_arg_embeddings = top_arg_span_embeddings.unsqueeze(1).repeat(1, num_pred_to_keep, 1, 1)
    span_pair_embeddings = torch.cat([expanded_pred_embeddings, expanded_arg_embeddings], dim=-1)
    return span_pair_embeddings

  def compute_srl_scores(self,
                         pairwise_embeddings: torch.FloatTensor,
                         top_pred_span_mention_scores: torch.FloatTensor,
                         top_arg_span_mention_scores: torch.FloatTensor) -> torch.FloatTensor:
    # Shape (batch_size, num_pred_to_keep, num_arg_to_keep, n_classes)
    scores = self.semantic_role_predictor(pairwise_embeddings)
    # top_pred_span_mention_scores = (batch_size, num_pred_to_keep, 1)
    num_pred_to_keep = top_pred_span_mention_scores.size(1)
    num_arg_to_keep = top_arg_span_mention_scores.size(1)
    expanded_top_pred_span_mention_scores = top_pred_span_mention_scores.unsqueeze(2).repeat(1, 1, num_arg_to_keep, 1)
    expanded_top_arg_span_mention_scores = top_arg_span_mention_scores.unsqueeze(1).repeat(1, num_pred_to_keep, 1, 1)
    dummy_scores = torch.zeros(scores.size()[:-1] + (1,)).to(pairwise_embeddings.device)
    scores = scores + expanded_top_arg_span_mention_scores + expanded_top_pred_span_mention_scores
    scores = torch.cat([dummy_scores, scores], dim=-1)
    return scores

















