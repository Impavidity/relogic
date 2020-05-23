from relogic.logickit.dataflow.dataflow import Example, Feature, MiniBatch, DataFlow
from transformers.tokenization_roberta import RobertaTokenizer
from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_bart import BartTokenizer
import torch
from relogic.logickit.utils import create_tensor_by_stacking, create_tensor
from typing import List
from collections import defaultdict

vocab_dict = {
  # "<": "is less than",
  # ">": "is more than",
  # "<=": "is not more than",
  # ">=": "is not less than",
  "avg": "average",
  "max": "maximum",
  "min": "minimum",
  # "sum": "total",
  "desc": "descending",
  "asc": "ascending"
}

class RerankingExample(Example):
  def __init__(self, idx, text, sql_candidates, entity_to_text, labels=None):
    super().__init__()
    self.idx = idx
    self.text = text
    self.sql_candidates = sql_candidates
    self.labels = labels
    self.entity_to_text = entity_to_text
    if len(sql_candidates) == 0:
      self.is_valid = False

  def process(self, tokenizers, *inputs, **kwargs):
    for tokenizer in tokenizers.values():
      if not isinstance(tokenizer, BartTokenizer) and not isinstance(tokenizer, RobertaTokenizer):
        continue
      add_prefix_space = isinstance(tokenizer, RobertaTokenizer)
      self.padding_ids = tokenizer.pad_token_id

      self.input_tokens = []
      self.input_token_spans = []
      start_idx = 1
      end_idx = 1
      for word in self.text.split():
        word_tokens = tokenizer.tokenize(word.lower(), add_prefix_space=add_prefix_space)
        if len(word_tokens) > 0:
          self.input_tokens.extend(word_tokens)
          end_idx += len(word_tokens)
          self.input_token_spans.append((start_idx, end_idx))
          start_idx = end_idx

      self.input_tokens = [tokenizer.cls_token] + self.input_tokens + [tokenizer.sep_token]
      self.input_token_ids = tokenizer.convert_tokens_to_ids(self.input_tokens)


      self.sql_candidates_token_list = []
      self.sql_candidates_token_ids_list = []
      self.sql_candidates_token_span_list = []
      for sql_candidate in self.sql_candidates:
        sql_tokens = []
        sql_token_spans = []
        start_idx = len(self.input_token_ids)
        end_idx = start_idx
        for word in sql_candidate.split():
          # if word in self.entity_to_text:
          #   word = self.entity_to_text[word]
          word = word.lower()
          # if word in vocab_dict:
          #   word = vocab_dict[word]
          word_tokens = tokenizer.tokenize(word, add_prefix_space=add_prefix_space)
          if len(word_tokens) > 0:
            sql_tokens.extend(word_tokens)
            end_idx += len(word_tokens)
            sql_token_spans.append((start_idx, end_idx))
            start_idx = end_idx
        self.sql_candidates_token_list.append(sql_tokens + [tokenizer.sep_token])
        self.sql_candidates_token_ids_list.append(tokenizer.convert_tokens_to_ids(self.sql_candidates_token_list[-1]))
        self.sql_candidates_token_span_list.append(sql_token_spans)

    if self.labels is not None:
      label_mapping = kwargs.pop("label_mapping", None)
      self.label_ids = []
      for label in self.labels:
        self.label_ids.append(label_mapping[label])

  @classmethod
  def from_structure(cls, structure):
    return cls(
      idx=structure.idx,
      text=structure.text,
      sql_candidates=structure.sql_candidates)


  @classmethod
  def from_json(cls, example):
    return cls(
      idx=example["idx"],
      text=example["text"],
      sql_candidates=example["sql_candidates"],
      entity_to_text=example["entity_to_text"],
      labels=example["labels"])

  @property
  def len(self):
    return len(self.input_token_ids)

class RerankingFeature(Feature):
  def __init__(self, *inputs, **kwargs):
    super().__init__()
    self.input_token_and_sql_candidate_token_ids_list = kwargs.pop("input_token_and_sql_candidate_token_ids_list")
    self.input_token_and_sql_candidate_token_attention_mask_list = kwargs.pop("input_token_and_sql_candidate_token_attention_mask_list")
    self.input_token_and_sql_candidate_token_segment_ids_list = kwargs.pop("input_token_and_sql_candidate_token_segment_ids_list")
    self.candidate_span = kwargs.pop("candidate_span")
    self.label_ids = kwargs.pop("label_ids")


class RerankingMiniBatch(MiniBatch):
  def __init__(self, *inputs, **kwargs):
    super().__init__(*inputs, **kwargs)

  def generate_input(self, device, use_label):
    inputs = {}
    inputs["task_name"] = self.task_name

    inputs["input_token_and_sql_candidate_token_ids"] = create_tensor_by_stacking(self.input_features,
            "input_token_and_sql_candidate_token_ids_list", torch.long, device)
    inputs["input_token_and_sql_candidate_token_attention_mask_list"] = create_tensor_by_stacking(
      self.input_features, "input_token_and_sql_candidate_token_attention_mask_list", torch.long, device)
    inputs["input_token_and_sql_candidate_token_segment_ids_list"] = create_tensor_by_stacking(
      self.input_features, "input_token_and_sql_candidate_token_segment_ids_list", torch.long, device)
    inputs["candidate_span"] = create_tensor(self.input_features, "candidate_span", torch.long, device)
    if use_label:
      inputs["label_ids"] = create_tensor_by_stacking(self.input_features,
              "label_ids", torch.long, device)
    else:
      inputs["label_ids"] = None
    return inputs


class RerankingDataFlow(DataFlow):
  def __init__(self, config, task_name, tokenizers, label_mapping):
    super().__init__(config, task_name, tokenizers, label_mapping)
    self.use_pair_loss = True

  @property
  def example_class(self):
    return RerankingExample

  @property
  def minibatch_class(self):
    return RerankingMiniBatch

  def process_example(self, example):
    example.process(
      tokenizers=self.tokenizers,
      label_mapping=self.label_mapping
    )

  def convert_examples_to_features(self, examples):
    examples: List[RerankingExample]
    features = []

    if not self.use_pair_loss:
      max_token_length = max(
        [len(example.input_tokens) + max([len(sql_candidates_tokens) for
           sql_candidates_tokens in example.sql_candidates_token_list]) for example in examples])

      start_idx = 0

      for idx, example in enumerate(examples):
        input_token_and_sql_candidate_token_ids_list = []
        input_token_and_sql_candidate_token_attention_mask_list = []
        input_token_and_sql_candidate_token_segment_ids_list = []
        for sql_candidates_token_ids in example.sql_candidates_token_ids_list:
          padding_length = max_token_length - len(sql_candidates_token_ids) - len(example.input_token_ids)
          input_token_and_sql_candidate_token_ids_list.append(
            example.input_token_ids + sql_candidates_token_ids + [example.padding_ids] * padding_length)
          input_token_and_sql_candidate_token_attention_mask_list.append(
            [1] * (max_token_length - padding_length) + [0] * padding_length
          )
          input_token_and_sql_candidate_token_segment_ids_list.append(
            [0] * len(example.input_token_ids) + [1] * len(sql_candidates_token_ids) + [0] * padding_length
          )
        end_idx = start_idx + len(example.sql_candidates_token_ids_list)

        features.append(RerankingFeature(
          input_token_and_sql_candidate_token_ids_list=input_token_and_sql_candidate_token_ids_list,
          input_token_and_sql_candidate_token_attention_mask_list=input_token_and_sql_candidate_token_attention_mask_list,
          input_token_and_sql_candidate_token_segment_ids_list=input_token_and_sql_candidate_token_segment_ids_list,
          candidate_span=(start_idx, end_idx),
          label_ids=example.label_ids))

        start_idx = end_idx

    else:
      # the feature creation is based on the label
      max_token_length = max([len(example.input_tokens) + 2 * max([len(sql_candidates_tokens) for
           sql_candidates_tokens in example.sql_candidates_token_list]) for example in examples])
      start_idx = 0
      for idx, example in enumerate(examples):
        input_token_and_sql_candidate_token_ids_list = []
        input_token_and_sql_candidate_token_attention_mask_list = []
        labels = []
        candidate_size = len(example.sql_candidates_token_ids_list)
        for i in range(candidate_size):
          for j in range(candidate_size):
            if i == j:
              continue
            left_candidate = example.sql_candidates_token_ids_list[i]
            right_candidate = example.sql_candidates_token_ids_list[j]
            labels.append(determine_label(example.label_ids[i], example.label_ids[j]))
            padding_length = max_token_length - len(left_candidate) - len(right_candidate) - len(example.input_token_ids)
            input_token_and_sql_candidate_token_ids_list.append(
              example.input_token_ids + left_candidate + right_candidate + [example.padding_ids] * padding_length)
            input_token_and_sql_candidate_token_attention_mask_list.append(
              [1] * (max_token_length - padding_length) + [0] * padding_length
            )
        end_idx = start_idx + len(example.sql_candidates_token_ids_list)
        features.append(RerankingFeature(
          input_token_and_sql_candidate_token_ids_list=input_token_and_sql_candidate_token_ids_list,
          input_token_and_sql_candidate_token_attention_mask_list=input_token_and_sql_candidate_token_attention_mask_list,
          input_token_and_sql_candidate_token_segment_ids_list=None,
          candidate_span=(start_idx, end_idx),
          label_ids=labels))


    return features

  def decode_to_labels(self, preds, mb):
    if not self.use_pair_loss:
      start_idx = 0
      pred_labels = []
      scores = []
      for example in mb.examples:
        end_idx = len(example.sql_candidates) + start_idx
        score_seg = preds["score"][start_idx: end_idx]
        pred_label = ["0"] * len(score_seg)
        pred_label[score_seg.argmax().item()] = "1"
        pred_labels.append(pred_label)
        scores.append(score_seg.data.cpu().tolist())
        start_idx = end_idx
      return pred_labels, scores
    else:
      start_idx = 0
      pred_labels = []
      scores = []
      for example in mb.examples:
        candidate_size = len(example.sql_candidates)
        span_size = candidate_size * (candidate_size -1)
        end_idx = span_size + start_idx
        score_seg = preds["score"][start_idx: end_idx]
        counter = defaultdict(int)
        idx = 0
        pred_label = ["0"] * candidate_size
        for i in range(candidate_size):
          for j in range(candidate_size):
            if i == j:
              continue
            if score_seg[idx].item() > 0.5:
              counter[j] += 1
            idx += 1
        if len(counter) > 0:
          selected_idx = list(sorted(counter.items(), key=lambda x: x[1], reverse=True))[0][0]
        else:
          selected_idx = 0
        pred_label[selected_idx] = "1"
        start_idx = end_idx
        pred_labels.append(pred_label)
        scores.append([0] * candidate_size)
      return pred_labels, scores


def determine_label(l_label, r_label):
  if l_label == 0 and r_label == 1:
    return 1
  return 0
