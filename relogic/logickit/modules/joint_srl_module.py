import torch.nn as nn
import torch
from relogic.logickit.modules.span_extractors.endpoint_span_extractor import EndpointSpanExtractor
from relogic.logickit.modules.span_extractors.self_attentive_span_extractor import SelfAttentiveSpanExtractor
from relogic.logickit.modules.span_extractors.average_span_extractor import AverageSpanExtractor
from relogic.logickit.modules.span_extractors.attentive_span_extractor import AttentiveSpanExtractor
from relogic.logickit.modules.pruner import Pruner
from relogic.logickit.inference.modeling import gelu
from relogic.logickit.utils import utils
from relogic.logickit.base.constants import SPAN_REPR_KENTON_LEE, SPAN_REPR_AVE_MAX, SPAN_REPR_AVE

try:
  from torch_geometric.nn import GCNConv

  class GraphNet(nn.Module):
    def __init__(self, feature_size, hidden_size):
      super(GraphNet, self).__init__()
      self.conv1 = GCNConv(feature_size, hidden_size)
      self.conv2 = GCNConv(hidden_size, hidden_size)

    def forward(self, x, edge_index):
      x = self.conv1(x, edge_index)
      x = torch.relu(x)
      x = torch.dropout(x, training=self.training)
      x = self.conv2(x, edge_index)
      return x
except:
  pass

class LayerNorm(nn.Module):
  def __init__(self, hidden_size, eps=1e-12):
    """Construct a layernorm module in the TF style (epsilon inside the square root).
    """
    super(LayerNorm, self).__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.bias = nn.Parameter(torch.zeros(hidden_size))
    self.variance_epsilon = eps

  def forward(self, x):
    u = x.mean(-1, keepdim=True)
    s = (x - u).pow(2).mean(-1, keepdim=True)
    x = (x - u) / torch.sqrt(s + self.variance_epsilon)
    return self.weight * x + self.bias

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

    hidden_size = config.hidden_size

    self.attentive_span_extractor = AttentiveSpanExtractor(input_dim=hidden_size)
    # SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
    # self.average_span_extractor = AverageSpanExtractor(input_dim=config.hidden_size)

    if self.config.srl_arg_span_repr == SPAN_REPR_AVE:
      arg_span_dim = hidden_size
    elif self.config.srl_arg_span_repr == SPAN_REPR_AVE_MAX:
      arg_span_dim = hidden_size * 2
    elif self.config.srl_arg_span_repr == SPAN_REPR_KENTON_LEE:
      arg_span_dim = hidden_size * 3 + config.span_width_embedding_dim
    else:
      raise ValueError("The span repr is not defined {}".format(self.config.srl_arg_span_repr))

    if self.config.srl_pred_span_repr == SPAN_REPR_AVE:
      pred_span_dim = hidden_size
    elif self.config.srl_pred_span_repr == SPAN_REPR_AVE_MAX:
      pred_span_dim = hidden_size * 2
    elif self.config.srl_pred_span_repr == SPAN_REPR_KENTON_LEE:
      pred_span_dim = hidden_size * 2
    else:
      raise ValueError("The span repr is not defined {}".format(self.config.srl_pred_span_repr))


    arg_feedforward_scorer = nn.Sequential(
      nn.Linear(in_features=arg_span_dim,
                out_features=config.hidden_size),
      nn.ReLU(),
      # LayerNorm(hidden_size=config.hidden_size),
      nn.Linear(in_features=config.hidden_size, out_features=1))
    self.mention_pruner = Pruner(arg_feedforward_scorer)
    predicate_feedforward_scorer = nn.Sequential(
      nn.Linear(in_features=pred_span_dim, out_features=config.hidden_size),
      nn.ReLU(),
      # LayerNorm(hidden_size=config.hidden_size),
      nn.Linear(in_features=config.hidden_size, out_features=1))
    self.predicate_pruner = Pruner(predicate_feedforward_scorer)

    self.semantic_role_predictor = nn.Sequential(
      nn.Linear(in_features=pred_span_dim + arg_span_dim + (300 if hasattr(config, "srl_use_label_embedding") and config.srl_use_label_embedding else 0),
                out_features=hidden_size),
      nn.ReLU(),
      # LayerNorm(hidden_size=config.hidden_size),
      nn.Linear(in_features=hidden_size, out_features=self.n_classes-1))
    self.constant_pred = nn.Parameter(torch.LongTensor([15]), requires_grad=False)
    self.constant_arg = nn.Parameter(torch.LongTensor([30]), requires_grad=False)
    # hard code here for now
    # TODO: make it to take the size from the vocab file
    if hasattr(config, "srl_use_label_embedding") and config.srl_use_label_embedding:
      self.label_embdding_key = nn.Embedding(27038, 300)
      self.label_embdding_value = nn.Embedding(27038, 300)
      self.pair_to_embedding = nn.Sequential(
        nn.Linear(in_features=arg_span_dim + pred_span_dim, out_features=config.hidden_size),
        nn.ReLU(),
        nn.Linear(in_features=config.hidden_size, out_features=300))
      self.to_attention = nn.Sequential(
        nn.Linear(in_features=600, out_features=100),
        nn.ReLU(),
        nn.Linear(in_features=100, out_features=1))
      # self.to_key = nn.Linear(in_features=300, out_features=300)
      # self.to_value = nn.Linear(in_features=300, out_features=300)
    # Hard code here for postag size
    if hasattr(config, "srl_compute_pos_tag_loss") and config.srl_compute_pos_tag_loss:
      self.to_pos_tag = nn.Sequential(
        nn.Linear(in_features=config.hidden_size, out_features=300),
        nn.ReLU(),
        nn.Linear(in_features=300, out_features=46)
      )
      # self.to_pos_tag = nn.Linear(in_features=config.hidden_size, out_features=46)

  def forward(self, *inputs, **kwargs):
    features = kwargs.pop("features")
    pos_tag_features = features[0]
    pred_features = features[1]
    arg_features = features[2]
    features = features[3]
    # TODO : quick fix for lstm testing
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
        sequence_tensor=features, value_tensor=arg_features, span_indices=arg_candidates, span_indices_mask=arg_mask)

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
    if self.config.srl_use_gold_argument:
      num_spans_to_keep = torch.sum(arg_mask, dim=-1).long()
    else:
      num_spans_to_keep = (torch.sum(arg_mask, dim=-1) * 0.8).long()
      num_spans_to_keep = torch.min(num_spans_to_keep, self.constant_arg) # quick fix

    (top_arg_span_embeddings, top_arg_span_mask,
     top_arg_span_indices, top_arg_span_mention_scores,
     arg_span_mention_full_scores) = self.mention_pruner(
      arg_embeddings, arg_mask, num_spans_to_keep)
    # top_arg_span_mask = (batch_size, max_arg_to_keep)

    if self.config.srl_pred_span_repr == SPAN_REPR_KENTON_LEE:
      pred_embeddings = self.endpoint_predicate_span_extractor(
        sequence_tensor=pred_features, span_indices=predicate_candidates, span_indices_mask=predicate_mask)
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

    if self.config.srl_use_gold_argument:
      num_preds_to_keep = (torch.sum(predicate_mask, dim=-1)).long()
    else:
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

    # Play with the label embeddings
    if "label_candidates" in kwargs and hasattr(self.config, "srl_use_label_embedding") and self.config.srl_use_label_embedding:
      label_candidates = kwargs["label_candidates"]
      label_candidates_mask = kwargs["label_candidates_mask"]

      label_candidate_embeddings_key = self.label_embdding_key(label_candidates)
      label_candidate_embeddings_value = self.label_embdding_value(label_candidates)
      # (batch, max_sent_length (without tokenization), max_label_candidate_size)

      top_pred_label_candidate_embeddings_key = utils.batched_index_select_tensor(label_candidate_embeddings_key,
                                 top_pred_span_indices,
                                 flat_top_pred_span_indices)
      top_pred_label_candidate_embeddings_value = utils.batched_index_select_tensor(label_candidate_embeddings_value,
                                 top_pred_span_indices,
                                 flat_top_pred_span_indices)

      top_pred_label_candidate_mask = utils.batched_index_select_tensor(label_candidates_mask,
                                  top_pred_span_indices,
                                  flat_top_pred_span_indices)
      # (batch, max_pred_candidate_size, max_label_candidate_size, embedding_size)
      max_label_candidate_size = label_candidates.size(2)
      max_argument_size = span_pair_embeddings.size(2)

      transformed_pair_embedding = self.pair_to_embedding(span_pair_embeddings)

      expanded_top_pred_label_candidate_embeddings_key = top_pred_label_candidate_embeddings_key.unsqueeze(2).repeat(
        1, 1, max_argument_size, 1, 1)
      expanded_top_pred_label_candidate_embeddings_value = top_pred_label_candidate_embeddings_value.unsqueeze(2).repeat(
        1, 1, max_argument_size, 1, 1)

      # Attention Method 1
      # expanded_transformed_span_pair_embeddings = transformed_pair_embedding.unsqueeze(3).repeat(
      #   1, 1, 1, max_label_candidate_size, 1)
      # concat = torch.cat([expanded_transformed_span_pair_embeddings, expanded_top_pred_label_candidate_embeddings_key], dim=-1)
      # attn_weights = self.to_attention(concat).squeeze(-1)

      # Attention Method 2
      # (batch_size, predicate_size, argument_size, dim=300)
      # (batch_size, predicate_size, arugment_size, label_size, dim=300)
      attn_weights = torch.matmul(transformed_pair_embedding.unsqueeze(3),
                                  expanded_top_pred_label_candidate_embeddings_key.transpose(-1, -2)).squeeze(3)
      # (batch_size, predicate_size, argument_size, label_size)

      expanded_top_pred_label_candidate_mask = top_pred_label_candidate_mask.unsqueeze(2).repeat(1, 1, max_argument_size, 1)

      attn_weights = utils.masked_softmax(attn_weights, expanded_top_pred_label_candidate_mask)
      # (batch, max_pred_cand, max_arg_cand, max_label_can)
      extra_feature = utils.weighted_sum(expanded_top_pred_label_candidate_embeddings_value, attn_weights)
      span_pair_embeddings = torch.cat([span_pair_embeddings, extra_feature], dim=-1)
      # (batch, max_pred_cand, )

    # if self.config.srl_pair_prune:
    #   matched_predicate_span = kwargs.pop("matched_predicate_span")
    #   # (batch, predicate_spans_num, 2)
    #   predicate_span_to_idx_dict = self.build_span_to_idx_dict(top_pred_spans)
    #   idx_to_collect_from_pair_embedding = []
    #   idx_to_collect_from_example_embedding = []
    #
    #
    #
    #   paired_mask = None
    #   num_pairs_to_keep = None
    #   top_pairs, top_pairs_mask, top_pairs_indices = self.pair_prune(span_pair_embeddings, paired_mask, num_pairs_to_keep)
    #   # top_pairs = (batch, predicate_size, argument_size, embedding_size)
    #   example_pairs = None
    #   # examples = (batch, predicate_size, argument_size', embedding_size)
    #   label_embeddings = None
    #   # label_embeddings = (batch, predicate_size, argument_size', label_embedding)
    #   attention_weights = None
    #   # attention_weights = (batch, predicate_size, argument_size, argument_size')
    #   # label_weighted_sum = attention_weights * label_embeddings = (batch, predicate_size, argument_size, label_embedding)
    #   # zeros = (batch, num_pred_to_keep, num_arg_to_keep, label_embedding_size)
    #   # expanded_label_repr = zeros + label_weighted_sum
    #   expanded_label_repr = None
    #   span_pair_embeddings = torch.cat([span_pair_embeddings, expanded_label_repr], dim=-1)

    """
    Joint Prediction with POS tag and predicate.
    We direct label the head word to see the performance.
    """
    if hasattr(self.config, "srl_compute_pos_tag_loss") and self.config.srl_compute_pos_tag_loss:
      pos_tag_logits = self.to_pos_tag(pos_tag_features)
    else:
      pos_tag_logits = None
    # We directly return to compute the loss

    srl_scores = self.compute_srl_scores(span_pair_embeddings,
                                         top_pred_span_mention_scores,
                                         top_arg_span_mention_scores)

    # Joint POS-Predicate Sequence labeling.

    return (srl_scores, top_pred_spans,
            top_arg_spans, top_pred_span_mask, top_arg_span_mask,
            pred_span_mention_full_scores, arg_span_mention_full_scores, pos_tag_logits)

  def build_span_to_idx_dict(self, spans: torch.LongTensor):
    """
    TODO: Push these operations to C++ if possible
    """
    span_to_idx_dict = {}
    for batch_idx in range(spans.size(0)):
      span_to_idx_dict[batch_idx] = {}
      for idx, span in enumerate(spans[batch_idx]):
        span_data = tuple(span.data.cpu().numpy())
        if span_data not in span_to_idx_dict[batch_idx]:
          span_to_idx_dict[batch_idx][span_data] = idx
    return span_to_idx_dict


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
