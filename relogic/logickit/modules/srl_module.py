import torch.nn as nn
import torch
from relogic.logickit.modules.rnn import dynamic_rnn

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


class SRLModule(nn.Module):
  def __init__(self, config, task_name, n_classes):
    super(SRLModule, self).__init__()
    self.config = config
    self.indicator_embedding = nn.Embedding(
      num_embeddings=2,
      embedding_dim=10)
    self.primary = PredictionModule(config, task_name, n_classes)

  def forward(self,
              input,
              view_type="primary",
              final_multi_head_repr=None,
              input_mask=None,
              segment_ids=None,
              extra_args=None):
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