import json
from typing import List, Tuple, Dict
import os

import torch

from relogic.logickit.dataflow import DataFlow, Example, Feature, MiniBatch
from relogic.logickit.tokenizer.tokenizer_roberta_xlm import RobertaXLMTokenizer
from relogic.logickit.utils import create_tensor, filter_head_prediction
from transformers.tokenization_utils import PreTrainedTokenizer
from relogic.logickit.tokenizer.fasttext_tokenization import FasttextTokenizer


MAX_SEQ_LENGTH=500

class SequenceExample(Example):
  """SequenceExample contains the attributes and functionality of an Sequence Labeling example.

  Args:
    text (str): A sentence string.
    labels (List[str]): A list of labels.

  History Log: Originally we use adapted tokenizer for determining the token head
      for the pretokenized sentence. Here we plan to change it with for loop.
      Then we do not need to adapt it for each tokenizer.
      ```
      self.text_tokens, self.text_is_head = tokenizer.tokenize(self.text)
      ```
      We will adapt the code from transformer example run_ner.py and utils_ner.py
  """
  def __init__(self, text, labels=None, lang=None):
    super(SequenceExample, self).__init__()
    self.text = text
    self.raw_tokens = text.split(" ")
    self.labels = labels
    self.label_padding = "_X_"
    self.lang = lang
    self.padding_id = 0

  def process(self, tokenizers: Dict, *inputs, **kwargs):
    """Process the Sequence Example..
    """
    for tokenizer in tokenizers.values():
      if isinstance(tokenizer, PreTrainedTokenizer):
        # BERT process part
        self.text_tokens = []
        self.text_is_head = []
        for word in self.raw_tokens:
          word_tokens = tokenizer.tokenize(word)
          if len(word_tokens) == 0:
            self.text_tokens.append("[UNK]")
            self.text_is_head.append(1)
          else:
            self.text_tokens.extend(word_tokens)
            self.text_is_head.extend([1] + [0] * (len(word_tokens) - 1))
        if len(self.text_tokens) > MAX_SEQ_LENGTH:
          self.text_tokens = self.text_tokens[:MAX_SEQ_LENGTH]
          self.text_is_head = self.text_is_head[:MAX_SEQ_LENGTH]

        self.tokens = ["[CLS]"] + self.text_tokens + ["[SEP]"]
        self.segment_ids = [0] * (len(self.tokens))
        self.is_head = [2] + self.text_is_head + [2]
        self.head_index = [idx for idx, value in enumerate(self.is_head) if value == 1]

        self.input_ids = tokenizer.convert_tokens_to_ids(self.tokens)
        self.input_mask = [1] * len(self.input_ids)

        if self.labels is not None:
          label_mapping = kwargs.get("label_mapping")
          self.label_padding_id = label_mapping[self.label_padding]
          self.label_ids = [self.label_padding_id] * len(self.input_ids)
          for idx, label in zip(self.head_index, self.labels):
            self.label_ids[idx] = label_mapping[label]
        else:
          self.label_ids = None

        if self.lang is not None:
          language_name2id = kwargs.get("language_name2id")
          if language_name2id is not None:
            self.lang_id = language_name2id[self.lang]
      if isinstance(tokenizer, RobertaXLMTokenizer):
        # RobertaXLM process part
        (self.tokens, self.is_head,
         self.input_ids) = tokenizer.tokenize_and_add_placeholder_and_convert_to_ids(self.text, self.raw_tokens)
        self.segment_ids = [0] * len(self.tokens)
        self.head_index = [idx for idx, value in enumerate(self.is_head) if value == 1]

        self.input_mask = [1] * len(self.input_ids)

        if self.labels is not None:
          label_mapping = kwargs.get("label_mapping")
          self.label_padding_id = label_mapping[self.label_padding]
          self.label_ids = [self.label_padding_id] * len(self.input_ids)
          for idx, label in zip(self.head_index, self.labels):
            self.label_ids[idx] = label_mapping[label]
        else:
          self.label_ids = None

        self.padding_id = 1
      if isinstance(tokenizer, FasttextTokenizer):
        self._text_tokens = tokenizer.tokenize(self.text)
        self._input_token_ids = tokenizer.convert_tokens_to_ids(self._text_tokens)
        self._input_token_mask = [True] * len(self._text_tokens)
        if self.labels is not None:
          label_mapping = kwargs.get("label_mapping")
          self.label_padding_id = label_mapping[self.label_padding]
          self._label_ids = [label_mapping[label] for label in self.labels]
        else:
          self._label_ids = None

  @classmethod
  def from_structure(cls, structure):
    return cls(text=structure.tokenized_text)

  @classmethod
  def from_json(cls, example):
    return cls(text=" ".join(example["tokens"]),
               labels=example.get("labels", None),
               lang=example.get("lang", None))

  @property
  def len(self):
    return len(self.input_ids)

  @property
  def _len(self):
    return len(self._input_token_ids)

class SequenceFeature(Feature):
  """Sequence Features.
  """
  def __init__(self, *inputs, **kwargs):
    super(SequenceFeature, self).__init__()
    self.input_ids = kwargs.pop("input_ids")
    self.input_mask = kwargs.pop("input_mask")
    self.segment_ids = kwargs.pop("segment_ids")
    self.is_head = kwargs.pop("is_head")
    self.label_ids = kwargs.pop("label_ids")
    self.lang_ids = kwargs.pop("lang_ids", None)

    # Classical Feature
    self._input_token_ids = kwargs.pop("_input_token_ids")
    self._token_length = kwargs.pop("_token_length")
    self._label_ids = kwargs.pop("_label_ids")
    self._input_token_mask = kwargs.pop("_input_token_mask")

class SequenceMiniBatch(MiniBatch):
  """

  """
  def __init__(self, *inputs, **kwargs):
    super(SequenceMiniBatch, self).__init__(*inputs, **kwargs)

  def generate_input(self, device, use_label):
    inputs = {}
    inputs["task_name"] = self.task_name
    inputs["input_ids"] = create_tensor(self.input_features, "input_ids",
                                        torch.long, device)
    inputs["input_mask"] = create_tensor(self.input_features, "input_mask",
                                         torch.long, device)
    inputs["segment_ids"] = create_tensor(self.input_features, "segment_ids",
                                          torch.long, device)
    inputs["input_head"] = create_tensor(self.input_features, "is_head",
                                         torch.long, device)
    if use_label:
      label_ids = create_tensor(self.input_features, "label_ids",
                                torch.long, device)
      inputs["label_ids"] = label_ids
    else:
      inputs["label_ids"] = None

    inputs["lang_ids"] = create_tensor(self.input_features, "lang_ids",
                                       torch.long, device)

    inputs["extra_args"] = {}
    if self.config.tasks[self.task_name]["selected_non_final_layers"] is not None:
      inputs["extra_args"]["selected_non_final_layers"] = self.config.tasks[self.task_name]["selected_non_final_layers"]

    # Classical Features
    inputs["_input_token_ids"] = create_tensor(self.input_features, "_input_token_ids",
                                               torch.long, device)
    inputs["_token_length"] = create_tensor(self.input_features, "_token_length",
                                            torch.long, device)

    # if inputs["_input_token_ids"] is not None:
    #   batch_size = inputs["_input_token_ids"].size(0)
    #   sequence_length = inputs["_input_token_ids"].size(1)
    #
    #   _tmp = torch.arange(0, sequence_length).long().expand(batch_size, sequence_length).to(device)
    #   inputs["_input_token_mask"] = _tmp < inputs["_token_length"].unsqueeze(1).expand_as(_tmp).contiguous()
    #   del _tmp
    # else:
    #   inputs["_input_token_mask"] = None
    inputs["_input_token_mask"] = create_tensor(self.input_features, "_input_token_mask",
                                                torch.bool, device)

    if use_label:
      inputs["_label_ids"] = create_tensor(self.input_features, "_label_ids",
                                           torch.long, device)
    else:
      inputs["_label_ids"] = None

    return inputs

class SequenceDataFlow(DataFlow):
  def __init__(self, config, task_name, tokenizers, label_mapping):
    super(SequenceDataFlow, self).__init__(config, task_name, tokenizers, label_mapping)
    self._inv_label_mapping = {v: k for k, v in label_mapping.items()}

  @property
  def example_class(self):
    return SequenceExample

  @property
  def minibatch_class(self):
    return SequenceMiniBatch

  def process_example(self, example: SequenceExample):
    example.process(tokenizers=self.tokenizers,
                    label_mapping=self.label_mapping,
                    language_name2id=None)
                    # self.config.language_name2id)
    # We defer the language_name2id process to the base config process

  def convert_examples_to_features(self, examples: List[SequenceExample]):
    examples: List[SequenceExample]
    features = []

    try:
      max_token_length = max([example.len for example in examples])
    except:
      max_token_length = None

    try:
      _max_token_length = max([example._len for example in examples])
    except:
      _max_token_length = None

    for idx, example in enumerate(examples):
      if max_token_length is not None:
        padding = [0] * (max_token_length - example.len)
        input_ids = example.input_ids + [example.padding_id] * (max_token_length - example.len)
        input_mask = example.input_mask + padding
        segment_ids = example.segment_ids + padding
        is_head = example.is_head + [2] * (max_token_length - example.len)

        if example.label_ids is not None:
          # We assume the label length is same as sequence length
          label_ids = example.label_ids + [example.label_padding_id] * (max_token_length - example.len)
        else:
          label_ids = None

        if example.lang is not None:
          # These code is for XLM
          lang = [example.lang_id] * max_token_length
        else:
          lang = None
      else:
        input_ids, input_mask, is_head, segment_ids, label_ids, lang = None, None, None, None, None, None

      if _max_token_length is not None:
        _input_token_ids = example._input_token_ids + [0] * (_max_token_length - example._len)
        _token_length = example._len
        _input_token_mask = example._input_token_mask + [False] * (_max_token_length - example._len)
        if example._label_ids is not None:
          _label_ids = example._label_ids + [example.label_padding_id] * (_max_token_length - example._len)
        else:
          _label_ids = None
      else:
        _input_token_ids, _token_length, _input_token_mask, _label_ids = None, None, None, None
      features.append(SequenceFeature(
        input_ids=input_ids,
        input_mask=input_mask,
        is_head=is_head,
        segment_ids=segment_ids,
        label_ids=label_ids,
        lang=lang,
        _input_token_ids=_input_token_ids,
        _token_length=_token_length,
        _input_token_mask=_input_token_mask,
        _label_ids=_label_ids))

    return features

  def decode_to_labels(self, preds, mb: MiniBatch):
    labels = []
    for example, pred_logits in zip(mb.examples, preds[mb.task_name]["logits"]):
      pred_label_index = pred_logits.argmax(-1).data.cpu().numpy()
      pred_label = [self._inv_label_mapping[y_pred] for y_pred in pred_label_index]
      is_head = example.is_head if hasattr(example, "is_head") else example._input_token_mask
      pred_label = filter_head_prediction(sentence_tags=pred_label, is_head=is_head)
      labels.append(pred_label)
    return labels



