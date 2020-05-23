import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_roberta import RobertaModel
from transformers.modeling_bert import BertModel
from transformers.modeling_bart import BartModel
from relogic.logickit.modules.contextualizers.self_attention import BertLayer
from relogic.logickit.modules.contextualizers.highway_lstm import HighwayLSTM

class Reranker(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    self.bert = RobertaModel.from_pretrained(config.bert_model)
    # self.bert.decoder = None
    # self.bert = self.bert.encoder
    #
    # self.ccv_feature_to_attention = nn.Sequential(
    #   nn.Linear(config.bert_hidden_size * 2, config.bert_hidden_size),
    #   nn.Tanh(),
    #   nn.Linear(config.bert_hidden_size, 1))
    self.to_logits = nn.Linear(config.bert_hidden_size, 2)
    # self.self_attention = BertLayer(
    #   hidden_size=config.bert_hidden_size,
    #   num_attention_heads=2,
    #   output_attentions=False,
    #   attention_probs_dropout_prob=0.1,
    #   layer_norm_eps=1e-12,
    #   hidden_dropout_prob=0.1,
    #   intermediate_size=512,
    #   hidden_act="gelu")
    # self.lstm = HighwayLSTM(num_layers=1, input_size=config.bert_hidden_size, hidden_size=int(config.bert_hidden_size /2), layer_dropout=0.1)

  # def cross_candidate_verification(self, features, candidate_span):
  #   max_span_length = (candidate_span[:, 1] - candidate_span[:, 0]).max().item()
  #   batch_size = candidate_span.size(0)
  #   dim = features.size(-1)
  #   feature_tensor = torch.zeros(batch_size, max_span_length, dim).to(features.device)
  #   attention_map = torch.zeros(batch_size, max_span_length).to(features.device)
  #   for i in range(batch_size):
  #     length = candidate_span[i][1] - candidate_span[i][0]
  #     feature_tensor[i, :length] = features[candidate_span[i][0]: candidate_span[i][1]]
  #     attention_map[i, :length] = 1
  #   concat_left = feature_tensor.unsqueeze(2).expand(
  #     batch_size, max_span_length, max_span_length, dim).contiguous().view(
  #     batch_size, max_span_length * max_span_length, dim)
  #   concat_right = feature_tensor.unsqueeze(1).expand(
  #     batch_size, max_span_length, max_span_length, dim).contiguous().view(
  #     batch_size, max_span_length * max_span_length, dim)
  #   # dot_product = torch.matmul(concat_left.unsqueeze(-2), concat_right.unsqueeze(-1)).squeeze(-1)
  #
  #   mask = torch.matmul(attention_map.unsqueeze(-1), attention_map.unsqueeze(-2))
  #
  #   r_ij = self.ccv_feature_to_attention(torch.cat([concat_left, concat_right], dim=-1))
  #   r_ij = r_ij.view(batch_size, max_span_length, max_span_length)
  #   r_i = (r_ij * mask).sum(-1).unsqueeze(-1)
  #   d_ij = r_ij / (r_i + 1e-10)
  #   v_j = torch.sigmoid((d_ij * mask).sum(-2))
  #
  #
  #   return v_j.view(-1)[attention_map.view(-1).bool()]


  def self_attention_transform(self, features, candidate_span):
    max_span_length = (candidate_span[:, 1] - candidate_span[:, 0]).max().item()
    batch_size = candidate_span.size(0)
    dim = features.size(-1)
    feature_tensor = torch.zeros(batch_size, max_span_length, dim).to(features.device)
    attention_map = torch.zeros(batch_size, max_span_length).to(features.device)
    for i in range(batch_size):
      length = candidate_span[i][1] - candidate_span[i][0]
      feature_tensor[i, :length] = features[candidate_span[i][0]: candidate_span[i][1]]
      attention_map[i, :length] = 1
    outputs = self.self_attention(hidden_states=feature_tensor, attention_mask=attention_map)
    v = outputs[0].view(-1, dim)[attention_map.view(-1).bool()]
    return v

  def contexualize(self, features, candidate_span):
    max_span_length = (candidate_span[:, 1] - candidate_span[:, 0]).max().item()
    batch_size = candidate_span.size(0)
    dim = features.size(-1)
    feature_tensor = torch.zeros(batch_size, max_span_length, dim).to(features.device)
    attention_map = torch.zeros(batch_size, max_span_length).to(features.device)
    for i in range(batch_size):
      length = candidate_span[i][1] - candidate_span[i][0]
      feature_tensor[i, :length] = features[candidate_span[i][0]: candidate_span[i][1]]
      attention_map[i, :length] = 1
    outputs, (_, _) = self.lstm(feature_tensor, attention_map.sum(-1))
    v = outputs.view(-1, dim)[attention_map.view(-1).bool()]
    return v


  def forward(self, *inputs, **kwargs):
    task_name = kwargs.pop("task_name")
    question_and_logic_form_token_ids = kwargs.pop("input_token_and_sql_candidate_token_ids")
    if question_and_logic_form_token_ids.size(0) == 0:
      return None
    question_and_logic_form_attention_mask = kwargs.pop("input_token_and_sql_candidate_token_attention_mask_list")
    # question_and_logic_form_segment_ids = kwargs.pop("input_token_and_sql_candidate_token_segment_ids_list")
    # candidate_span = kwargs.pop("candidate_span")
    bert_features = self.bert(input_ids=question_and_logic_form_token_ids,
              attention_mask=question_and_logic_form_attention_mask)
    global_feature = bert_features[0][:, 0]
    # v = self.cross_candidate_verification(global_feature, candidate_span)
    # v = self.self_attention_transform(global_feature, candidate_span)
    # v = self.contexualize(global_feature, candidate_span)

    score = self.to_logits(global_feature)

    # (batch_size, dim)

    # score = self.to_logits(bert_features[0][:, 0])

    # agg_score = 0.5 * score + 0.5 * v


    if self.training:
      label_ids = kwargs.pop("label_ids")
      results = {
        task_name: {
          "loss": F.cross_entropy(score, label_ids)
        }
      }
      return results
    else:
      results = {
        task_name: {
          "score": torch.softmax(score, dim=-1)[:, 1]
        }
      }
    return results