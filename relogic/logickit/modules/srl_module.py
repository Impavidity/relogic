import torch.nn as nn
import torch
from relogic.logickit.modules.rnn import dynamic_rnn
from relogic.logickit.modules.span_gcn import SpanGCNModule
from relogic.logickit.modules.span_gcn import select_span
import numpy as np

class PredictionModule(nn.Module):
  def __init__(self, config, task_name, n_classes, activate=True):
    super(PredictionModule, self).__init__()
    self.config = config
    self.task_name = task_name
    self.n_classes = n_classes
    if config.use_bilstm:
      self.bilstm = nn.LSTM(
        input_size=config.hidden_size + 10, # for indicator embedding
        hidden_size=config.hidden_size,
        bidirectional=True,
        batch_first=True,
        num_layers=1)
      self.projection = nn.Linear(4 * config.hidden_size, config.hidden_size)
    else:
      self.projection = nn.Linear(2 * (config.hidden_size + 10), config.hidden_size)
    self.activate = activate
    if activate:
      self.activation = nn.SELU()
    self.to_logits = nn.Linear(config.hidden_size, self.n_classes)
    # self.apply(self.init_weights)

  def init_weights(self, module):
    if isinstance(module, nn.Linear):
      module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    if isinstance(module, nn.Linear) and module.bias is not None:
      module.bias.data.zero_()

  def forward(self, input, indicator_embed, input_lengths, extra_args):
    # hidden = self.bilstm(input)
    input_concat = torch.cat([input, indicator_embed], dim=-1)
    if self.config.use_bilstm:
      output, (ht, ct) = dynamic_rnn(self.bilstm, input_concat, input_lengths)
    else:
      output = input_concat
    is_predicate_mask = (extra_args["is_predicate_id"] == 1)[:, :output.size(1)]
    concat = torch.cat([output, output[is_predicate_mask].unsqueeze(1).repeat(1, output.size(1), 1)], dim=-1)
    # print(input[is_predicate_mask].size())
    projected = self.projection(concat)
    if self.activate:
      projected = self.activation(projected)
    logits = self.to_logits(projected)
    return logits

class DescriptionAware(nn.Module):
  def __init__(self, config, task_name, n_classes, activate=True):
    super(DescriptionAware, self).__init__()
    self.config = config
    self.n_classes = n_classes
    self.label_embedding = nn.Embedding(self.n_classes, self.config.label_embed_dim)
    # self.boundary_embedding = nn.Embedding(self.config.boundary_embed_num, self.config.boundary_embed_dim)
    self.word_embedding = nn.Embedding(self.config.external_vocab_size, self.config.external_vocab_embed_size)
    self.word_embedding.weight.data.copy_(torch.from_numpy(np.load(config.external_embeddings)))
    print("Loading embedding from {}".format(config.external_embeddings))
    self.padding = nn.Parameter(torch.zeros(config.hidden_size), requires_grad=False)
    self.ones = nn.Parameter(torch.ones(1, 1), requires_grad=False)
    self.attention_weight_1 = nn.Linear(config.hidden_size + self.config.external_vocab_embed_dim, config.hidden_size)
    self.attention_weight_2 = nn.Linear(config.hidden_size, 1)
    self.to_logits_1 = nn.Linear(config.hidden_size * 2 + config.label_embed_dim + config.external_vocab_embed_dim, 300)
    self.to_logits_2 = nn.Linear(300, 1)
    # self.to_logits_2 = nn.Linear(config.hidden_size + self.config.external_vocab_embed_size, self.n_classes)
    # self.to_label_size = nn.Linear(n_classes, n_classes)
    self.to_logits_3 = nn.Linear(config.hidden_size, n_classes)

  def forward(self, input, predicate_span, predicate_descriptions_ids=None,
              argument_descriptions_ids=None):
    """

    :param input:
    :param predicate_idx:
    :param predicate_descriptions:
    :return:
    """

    predicate_start_index, predicate_end_index = predicate_span
    predicate_sense_num = predicate_descriptions_ids.size(1)
    predicate_hidden = select_span(input, predicate_start_index, predicate_end_index, self.padding)
    predicate_hidden_agg = self.agg(predicate_hidden, predicate_end_index - predicate_start_index)
    # (batch, dim)
    
    # predicate description
    
    # predicate_descriptions = (batch, predicate_sense, predicate_description)
    predicate_description_embed = self.word_embedding(predicate_descriptions_ids)

    predicate_descriptions_mask = (predicate_descriptions_ids > 0)
    predicate_descriptions_lengths = predicate_descriptions_mask.sum(-1)
    predicate_sense_length_mask  = (1 - (predicate_descriptions_lengths > 0).long()) * -100000
    # lengths = (batch, predicate_sense, 1)

    predicate_description_agg = self.agg(
      predicate_description_embed,
      predicate_descriptions_lengths,
      mask=predicate_descriptions_mask)

    predicate_hidden_expanded = predicate_hidden_agg.unsqueeze(1).repeat(1, predicate_sense_num, 1)
    predicate_descriptions_weights, predicate_descriptions_weighted_sum = self.attention_module(
      predicate_hidden_expanded, predicate_description_agg, attention_mask=predicate_sense_length_mask)

    # (batch, predicate_sense, 1) (batch, dim)
    
    # argument description

    # go-01 : A1: a1_desc, A2: a2_desc
    # go-02 : A1: a1_desc, A2: a2_desc
    # go -> go-01 0.7 , go-02 0.3
    # A1: 0.7 * go-01-A1 + 0.3 * go-02-A1
    # argument descriptions = (batch, predicate_sense, argument_type, argument_description)
    argument_descriptions_embed = self.word_embedding(argument_descriptions_ids)
    argument_descriptions_mask = (argument_descriptions_ids > 0)
    argument_descriptions_lengths = argument_descriptions_mask.sum(-1)
    # lengths = (batch, predicate_sense, argument_type, lengths)
    # argument descriptions embed = (batch, predicate_sense, argument_type, argument_description, dim)
    argument_descriptions_agg = self.agg(
      argument_descriptions_embed,
      argument_descriptions_lengths,
      mask=argument_descriptions_mask).transpose(1, 2)
    #
    # argument_descriptions_agg = (batch, argument_type, predicate_sense, dim)
    expanded_weight = predicate_descriptions_weights.unsqueeze(1).repeat(1, argument_descriptions_agg.size(1), 1, 1)
    # weights = (batch, predicate_sense, 1) ->
    # weights = (batch, argument_type, predicate_sense, 1)
    argument_descriptions_weighted_sum = torch.sum(expanded_weight * argument_descriptions_agg, dim=-2)
    # (batch, argument_type, dim)
    # label_embedding = (label_size, dim)
    label_info = torch.cat([
      self.label_embedding.weight.repeat(argument_descriptions_weighted_sum.size(0), 1, 1),
      argument_descriptions_weighted_sum], dim=-1)
    # label_info = self.label_embedding.weight.repeat(argument_descriptions_weighted_sum.size(0), 1, 1)
    # (batch, argument_type, dim)
    sentence_length = input.size(1)
    expanded_input = input.unsqueeze(-2).repeat(1, 1, self.n_classes, 1)
    # (batch, sentence, dim) -> (batch, sentence, 1, dim) -> (batch, sentence, label_size, dim)
    expanded_label_info = label_info.unsqueeze(1).repeat(1, sentence_length, 1, 1)
    # (batch, argument_type, dim) -> (batch, 1, argument_type(label_size), dim) -> (batch, sentence, argument_type, dim)
    # predicate_hidden_agg = (batch, dim)
    hidden = torch.cat([expanded_input, expanded_label_info,
                        predicate_hidden_agg.unsqueeze(1).unsqueeze(1).repeat(1, sentence_length, self.n_classes, 1)], dim=-1)
    # (batch, sentence, argument_type, dim) -> (batch, sentence, argument_type, 1) ->  (batch, sentence_length, argument_type)
    logits = self.to_logits_2(torch.relu(self.to_logits_1(hidden))).squeeze(-1)
    # print(logits)
    # logits = self.to_label_size(logits)

    # predicate_descriptions_weighted_sum_expanded = predicate_descriptions_weighted_sum.unsqueeze(1).repeat(1, sentence_length, 1)
    # (batch, sentence, dim)

    # logits = self.to_logits_3(input)
    return logits




  def agg(self, embed, lengths, mask=None):
    lengths = lengths.unsqueeze(-1).float()
    ones = torch.ones_like(lengths)
    if mask is None:
      return torch.sum(embed, -2) / torch.max(ones, lengths)
    else:
      return torch.sum(embed * mask.unsqueeze(-1).float(), -2) / torch.max(ones, lengths)

  def attention_module(self, predicate_hidden, predicate_description, attention_mask=None):
    concated = torch.cat([predicate_hidden, predicate_description], dim=2)
    weights = self.attention_weight_2(torch.relu(self.attention_weight_1(concated)))

    if attention_mask is not None:
      weights = weights + attention_mask.unsqueeze(-1).to(dtype=next(self.parameters()).dtype)

    weights = torch.softmax(weights, dim=1)

    # (batch, predicate_sense, 1)
    weighted_sum = torch.sum(weights * predicate_description, dim=1)
    return weights, weighted_sum


class PredicateDetection(nn.Module):
  def __init__(self, config):
    pass

  def forward(self, input):
    pass

class SRLModule(nn.Module):
  def __init__(self, config, task_name, n_classes):
    super(SRLModule, self).__init__()
    self.config = config
    if self.config.srl_module_type == "sequence_labeling":
      self.indicator_embedding = nn.Embedding(
        num_embeddings=2,
        embedding_dim=10)
      self.primary = PredictionModule(config, task_name, n_classes)
    elif self.config.srl_module_type == "span_gcn":
      self.span_gcn = SpanGCNModule(config, task_name, label_n_classes=n_classes)
    elif self.config.srl_module_type == "description_aware":
      self.description_aware = DescriptionAware(config, task_name, n_classes=n_classes)
    else:
      raise ValueError("SRL module type {} is wrong".format(self.config.srl_module_type))

  def forward(self,
              input,
              view_type="primary",
              final_multi_head_repr=None,
              input_mask=None,
              segment_ids=None,
              extra_args=None,
              **kwargs):
    if self.config.srl_module_type == "sequence_labeling":
      if view_type == "primary":
        input_lengths = (segment_ids == 0).sum(-1)
        # Including two place holder [CLS] [SEP]
        # lemma_embed = self.lemma_embedding(extra_args["predicate_lemma_id"])
        # (batch, 1, dim)
        # lemma_expand = lemma_embed.repeat(1, input.size(-2), 1)
        indicator_embed = self.indicator_embedding(extra_args["is_predicate_id"])
        primary_logits = self.primary(input, indicator_embed, input_lengths, extra_args)
        return primary_logits
      else:
        raise NotImplementedError("Partial View is not implemented for SRL Module")
    elif self.config.srl_module_type == "span_gcn":
      # "span_candidates" should be in extra_args for now
      # basically extract span information, and then use GCN for global inference
      # input, predicate_hidden, input_mask, predicate_length, bio_hidden=None, span_candidates=None, extra_args=None
      predicate_logits = None
      label_logits = self.span_gcn(input=input,
                    predicate_span=extra_args["predicate_span"],
                    span_candidates=extra_args["span_candidates"])
      return extra_args["span_candidates"], label_logits
      # We use provided span candidates for now
      # TODO: auto span detection
    elif self.config.srl_module_type == "description_aware":
      label_logits = self.description_aware(input=input,
            predicate_span=extra_args["predicate_span"],
            predicate_descriptions_ids=extra_args["predicate_descriptions_ids"],
            argument_descriptions_ids=extra_args["argument_descriptions_ids"])
      return label_logits

