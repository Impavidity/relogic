import torch.nn as nn
import torch
from relogic.logickit.modules.rnn import dynamic_rnn
import torch.nn.functional as F
from relogic.logickit.modules.span_gcn import select_span

class PredictionModule(nn.Module):
  def __init__(self, config, task_name, n_classes):
    super(PredictionModule, self).__init__()
    self.config = config
    self.task_name = task_name
    self.n_classes = n_classes
    if config.use_bilstm:
      self.bilstm = nn.LSTM(
        input_size=config.hidden_size, # for indicator embedding
        hidden_size=config.hidden_size,
        bidirectional=False,
        batch_first=True,
        num_layers=1)
      self.projection = nn.Linear(config.hidden_size, config.hidden_size)
      self.activation = nn.SELU()
      self.attention_W = nn.Linear(2 * config.hidden_size + 20, config.hidden_size, bias=False)
      self.attention_V = nn.Linear(config.hidden_size, 1, bias=False)
    if self.config.rel_extraction_module_type == "entity_span":
      hidden_size = 2 * config.hidden_size
    elif self.config.rel_extraction_module_type == "sent_repr":
      hidden_size = config.hidden_size
    elif self.config.rel_extraction_module_type == "hybrid":
      hidden_size = 3 * config.hidden_size
    else:
      raise ValueError("rel_extraction_module_type is not in [entity_span, sent_repr, hybrid], {}".format(
        self.config.rel_extraction_module_type))
    self.to_logits = nn.Linear(hidden_size, self.n_classes)
    if self.config.rel_extraction_module_type == "hybrid":
      self.padding = nn.Parameter(torch.zeros(config.hidden_size), requires_grad=False)

  def attention_module(self, query, candidate, mask):
    # query = (batch, dim), candidate = (batch, length, dim)
    query_duplicate = query.unsqueeze(1).expand(-1, candidate.size(1), -1)
    mask = mask.to(dtype=next(self.parameters()).dtype)
    u = self.attention_V(
      F.tanh(
        self.attention_W(torch.cat([query_duplicate, candidate], dim=-1))
      )).squeeze(-1) + (1.0-mask) * (-10000.0)
    # u = (batch, length)
    a = F.softmax(u, dim=-1)
    return a

  def forward(self, input, subj_indicator_embed, obj_indicator_embed, input_lengths, input_mask, extra_args, **kwargs):
    if self.config.use_bilstm:
      # input_concat = torch.cat([input, subj_indicator_embed, obj_indicator_embed], dim=-1)
      output, (ht, ct) = dynamic_rnn(self.bilstm, input, input_lengths)
      # ht (2, batch, dim)
      ht = ht.transpose(0, 1).contiguous().view(input.size(0), -1)
      # (batch_size, dim)
      prob = self.attention_module(
        query=ht,
        candidate=torch.cat([output, subj_indicator_embed, obj_indicator_embed], dim=-1),
        mask=input_mask)
      projected = (output * prob.unsqueeze(-1)).sum(1)
      # output (batch, sequence, dim) -> projected (batch, dim)
      projected = self.projection(projected)
      projected = self.activation(projected)
    else:
      if self.config.rel_extraction_module_type == "entity_span":
        if "start_of_subject" in extra_args and "start_of_object" in extra_args:
          start_of_subject = extra_args["start_of_subject"]
          start_of_object = extra_args["start_of_object"]
          start_of_subject_repr = input[torch.arange(input.size(0)), start_of_subject]
          start_of_object_repr = input[torch.arange(input.size(0)), start_of_object]
          projected = torch.cat([start_of_subject_repr, start_of_object_repr], dim=1)
          # logits = self.to_logits(torch.cat([start_of_subject_repr, start_of_object_repr], dim=1))
        else:
          raise ValueError("Start of Subject is not in extra args")
      elif self.config.rel_extraction_module_type == "sent_repr":
        projected = input[:, 0]
      elif self.config.rel_extraction_module_type == "hybrid":
        if ("start_of_subject" in extra_args and
              "start_of_object" in extra_args and
              "end_of_subject" in extra_args and
              "end_of_object" in extra_args):
          cls_hidden = input[:, 0]
          start_of_subject = extra_args["start_of_subject"]
          start_of_object = extra_args["start_of_object"]
          end_of_subject = extra_args["end_of_subject"]
          end_of_object = extra_args["end_of_object"]
          subject_repr = select_span(input, start_index=start_of_subject, end_index=end_of_subject, padding=self.padding)
          object_repr = select_span(input, start_index=start_of_object, end_index=end_of_object, padding=self.padding)
          subject_repr = self.agg(subject_repr, end_of_subject-start_of_subject)
          object_repr = self.agg(object_repr, end_of_object-start_of_object)
          projected = torch.cat([cls_hidden, subject_repr, object_repr], dim=-1)
        else:
          raise ValueError("Start of Subject is not in extra args")
      else:
        raise ValueError("rel_extraction_module_type is not in [entity_span, sent_repr, hybrid], {}".format(
          self.config.rel_extraction_module_type))
    logits = self.to_logits(projected)
    return logits

  def agg(self, embed, lengths):
    return torch.sum(embed, -2) / lengths.unsqueeze(1).float()



class RelExtractionModule(nn.Module):
  def __init__(self, config, task_name, n_classes):
    super(RelExtractionModule, self).__init__()
    self.config = config
    self.indicator_embedding = nn.Embedding(self.config.max_seq_length * 2 + 1, 10)

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
      subj_ind = extra_args['subj_indicator'] + self.config.max_seq_length # torch.tensor(self.config.max_seq_length, dtype=torch.long).to(input.device)
      obj_ind = extra_args['obj_indicator'] + self.config.max_seq_length # torch.tensor(self.config.max_seq_length, dtype=torch.long).to(input.device)
      subj_indicator_embed = self.indicator_embedding(subj_ind.unsqueeze(1))
      obj_indicator_embed = self.indicator_embedding(obj_ind.unsqueeze(1))
      primary_logits = self.primary(input, subj_indicator_embed, obj_indicator_embed, input_lengths, input_mask, extra_args)
      return primary_logits
    else:
      raise NotImplementedError("Partial View is not implemented")