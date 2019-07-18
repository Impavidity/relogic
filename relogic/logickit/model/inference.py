import torch
import torch.nn as nn
import torch.nn.functional as F

from relogic.logickit.model.encoder import Encoder
from relogic.logickit.model.branching_encoder import BranchingBertModel
from relogic.logickit.base import utils
from torch.nn import MarginRankingLoss, CrossEntropyLoss, KLDivLoss

from relogic.logickit.base.constants import *


class Inference(nn.Module):
  def __init__(self, config, tasks):
    super(Inference, self).__init__()
    self.config = config
    if config.branching_encoder:
      utils.log("Build Branching Bert Encoder")
      self.encoder = BranchingBertModel.from_pretrained(config.bert_model, encoder_structure=config.branching_structure)
    else:
      utils.log("Build Bert Encoder")
      self.encoder = Encoder.from_pretrained(config.bert_model)
    utils.log("Build Task Modules")
    self.tasks = nn.ModuleDict()
    for task in tasks:
      self.tasks.update([(task.name, task.get_module())])
    self.loss_fct = CrossEntropyLoss()
    self.loss_kl = KLDivLoss(reduction="batchmean")
    self.softmax = nn.Softmax(-1)

  def forward(self, task_name, input_ids, input_mask, input_head, segment_ids, label_ids, extra_args):
    features = self.encoder(
      input_ids=input_ids,
      token_type_ids=segment_ids,
      attention_mask=input_mask,
      output_all_encoded_layers=False,
      token_level_attention_mask=extra_args.get("token_level_attention_mask", None),
      route_path=extra_args.get("route_path", None),
      no_dropout=task_name in READING_COMPREHENSION_TASKS)
    if self.config.output_attentions:
      features, attention_map = features

    # The feature is a list from all layers
    if task_name != "unlabeled":
      logits = self.tasks[task_name](features, input_mask=input_mask, segment_ids=segment_ids, extra_args=extra_args)
      # For span extraction (Reading Comprehension), 
      # the label_ids is start_position and end_position, the logits is start_logits and end_logits
      # Need to adapt the following closs calculation somehow
      # Use the final hidden layer
      if label_ids is not None:
        if input_head is not None and task_name in ["er", "ner", "srl", "srl_conll05", "srl_conll09", "srl_conll12", "predicate_sense"]:
          active_loss = input_head[:, :logits.size(1)].contiguous().view(-1) == 1
          # I use logits.size(1) to get the label length. It is OK to filter extra things
          active_logits = logits.view(-1, logits.size(-1))[active_loss]
          active_labels = label_ids[:, :logits.size(1)].contiguous().view(-1)[active_loss]
          loss = self.loss_fct(active_logits, active_labels)
        elif task_name in READING_COMPREHENSION_TASKS:
          start_logits, end_logits = logits
          # start_logits, end_logits (batch, sentence_length)
          ignored_index = start_logits.size(1)
          start_positions, end_positions = label_ids.split(1, dim=-1)
          if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
          if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
          start_positions.clamp_(0, ignored_index)
          end_positions.clamp_(0, ignored_index)
          start_loss = F.cross_entropy(input=start_logits, target=start_positions, ignore_index=ignored_index)
          end_loss = F.cross_entropy(input=end_logits, target=end_positions, ignore_index=ignored_index)
          loss = (start_loss + end_loss) / 2
        else:
          loss = self.loss_fct(logits.view(-1, logits.size(-1)), label_ids.view(-1))
        return loss
      else:
        if self.config.output_attentions:
          return logits, attention_map
        else:
          return logits
    # else:
    #   results = {}
    #   # Teacher
    #   if label_ids is None:
    #     for task, task_module in self.tasks.items():
    #       results[task] = task_module(features[-1], input_mask=input_mask, extra_args=extra_args).detach()
    #       # Use final hidden layer
    #     return results
    #   else:
    #     # Partial
    #     for task, task_module in self.tasks.items():
    #       results[task] = task_module(
    #         features[-1].detach(), view_type="partial", final_multi_head_repr=final_multi_head_repr, input_mask=input_mask, segment_ids=segment_ids, extra_args=extra_args)
    #     loss = 0
    #     for task in self.tasks:
    #       for partial_view in results[task]:
    #         if input_head is not None:
    #           active_loss = input_head.view(-1) == 1
    #           active_logits = self.softmax(partial_view).view(-1, partial_view.size(-1))[active_loss]
    #           active_labels_logtis = self.softmax(label_ids[task]).view(-1, label_ids[task].size(-1))[active_loss]
    #           loss += self.loss_kl(active_logits.log(), active_labels_logtis)
    #         else:
    #           loss += self.loss_kl(self.softmax(partial_view).log(), self.softmax(label_ids[task]))
    #     return loss

class PairMatching(nn.Module):
  def __init__(self, config, tasks):
    super(PairMatching, self).__init__()
    utils.log("Build Bert Encoder")
    self.encoder = Encoder.from_pretrained(config.bert_model)
    utils.log("Build Task Modules")
    self.tasks = nn.ModuleDict()
    for task in tasks:
      self.tasks.update([(task.name, task.get_module())])
    self.loss_max_margin = MarginRankingLoss(margin=config.max_margin)
    self.distance = nn.PairwiseDistance(p=1)

  def forward(self, task_name,
              a_input_ids, a_input_mask, a_input_head, a_segment_ids,
              b_input_ids, b_input_mask, b_input_head, b_segment_ids,
              c_input_ids, c_input_mask, c_input_head, c_segment_ids, extra_args):
    a_features = self.encoder(
      input_ids=a_input_ids,
      token_type_ids=a_segment_ids,
      attention_mask=a_input_mask,
      output_all_encoded_layers=False,
      output_final_multi_head_repr=False)
    b_features = self.encoder(
      input_ids=b_input_ids,
      token_type_ids=b_segment_ids,
      attention_mask=b_input_mask,
      output_all_encoded_layers=False,
      output_final_multi_head_repr=False)
    c_features = self.encoder(
      input_ids=c_input_ids,
      token_type_ids=c_segment_ids,
      attention_mask=c_input_mask,
      output_all_encoded_layers=False,
      output_final_multi_head_repr=False)
    a_logits = self.tasks[task_name](a_features, input_mask=a_input_mask, segment_ids=a_segment_ids, extra_args=extra_args)
    b_logits = self.tasks[task_name](b_features, input_mask=b_input_mask, segment_ids=b_segment_ids, extra_args=extra_args)
    c_logits = self.tasks[task_name](c_features, input_mask=c_input_mask, segment_ids=c_segment_ids, extra_args=extra_args)
    positive_distance = self.distance(a_logits, b_logits)
    negative_distance = self.distance(a_logits, c_logits)
    loss = self.loss_max_margin(positive_distance, negative_distance, target=extra_args["target"])
    return loss

  def gen_repr(self, task_name, input_ids, input_mask, input_head, segment_ids, extra_args):
    features = self.encoder(
      input_ids=input_ids,
      token_type_ids=segment_ids,
      attention_mask=input_mask,
      output_all_encoded_layers=False,
      output_final_multi_head_repr=False)
    logits = self.tasks[task_name](features, input_mask=input_mask, segment_ids=segment_ids, extra_args=extra_args)
    return logits