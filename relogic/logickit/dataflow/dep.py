from relogic.logickit.dataflow import DataFlow, Example, Feature, MiniBatch
from typing import List, Dict, Tuple
import torch
from relogic.logickit.utils import create_tensor

from relogic.logickit.tokenizer.tokenization import BertTokenizer

class DependencyParsingExample(Example):
  """DependencyParsingExample


  """
  def __init__(self, text, arcs=None, labels=None, lang=None):
    super(DependencyParsingExample, self).__init__()
    self.text = text
    self.raw_tokens = text.split()
    self.arcs = arcs
    self.labels = labels
    self.lang = lang
    self.label_padding = "X"

  def process(self, tokenizers: Dict, *inputs, **kwargs):
    for tokenizer in tokenizers.values():
      if isinstance(tokenizer, BertTokenizer):
        self.text_tokens, self.text_is_head = tokenizer.tokenize(self.text)

        self.tokens = ["[CLS]"] + self.text_tokens + ["[SEP]"]
        self.segment_ids = [0] * (len(self.tokens))
        self.is_head = [2] + self.text_is_head + [2]
        self.head_index = [idx for idx, value in enumerate(self.is_head) if value == 1]

        self.input_ids = tokenizer.convert_tokens_to_ids(self.tokens)
        self.input_mask = [1] * len(self.input_ids)

        if self.arcs is not None and self.labels is not None:
          label_mapping = kwargs.get("label_mapping")
          self.label_padding_id = label_mapping[self.label_padding]
          self.label_ids = [self.label_padding_id] * len(self.input_ids)
          self.arcs_ids = [-1] * len(self.input_ids)
          assert(len(self.labels) == len(self.arcs))
          for idx, label, arc in zip(self.head_index, self.labels, self.arcs):
            self.label_ids[idx] = label_mapping[label]
            self.arcs_ids[idx] = arc
        else:
          self.label_ids = None
          self.arcs_ids = None

        if self.lang is not None:
          language_name2id = kwargs.get("language_name2id")
          if language_name2id is not None:
            self.lang_id = language_name2id[self.lang]

  @classmethod
  def from_structure(cls, structure):
    return cls(text=structure.text)

  @classmethod
  def from_json(cls, example):
    return cls(text=" ".join(example["tokens"]),
               arcs=example.get("arcs", None),
               labels=example.get("labels", None),
               lang=example.get("lang", None))

  @property
  def len(self):
      return len(self.input_ids)


class DependencyParsingFeature(Feature):
  """
  Sequence Features
  """
  def __init__(self, *inputs, **kwargs):
    super().__init__()
    self.input_ids = kwargs.pop("input_ids")
    self.input_mask = kwargs.pop("input_mask")
    self.segment_ids = kwargs.pop("segment_ids")
    self.is_head = kwargs.pop("is_head")
    self.arcs_ids = kwargs.pop("arcs_ids")
    self.label_ids = kwargs.pop("label_ids")
    self.lang_ids = kwargs.pop("lang_ids", None)

class DependencyParsingMiniBatch(MiniBatch):
  def __init__(self, *inputs, **kwargs):
    super().__init__(*inputs, **kwargs)

  def generate_input(self, device, use_label):
    inputs = {}
    inputs["task_name"] = self.task_name
    inputs["input_ids"] = create_tensor(self.input_features, "input_ids", torch.long, device)
    inputs["input_mask"] = create_tensor(self.input_features, "input_mask", torch.long, device)
    inputs["segment_ids"] = create_tensor(self.input_features, "segment_ids", torch.long, device)
    inputs["input_head"] = create_tensor(self.input_features, "is_head", torch.long, device)

    if use_label:
      label_ids = create_tensor(self.input_features, "label_ids", torch.long, device)
      inputs["label_ids"] = label_ids
      arcs_ids = create_tensor(self.input_features, "arcs_ids", torch.long, device)
      inputs["arcs_ids"] = arcs_ids
    else:
      inputs["label_ids"] = None
      inputs["arcs_ids"] = None

    inputs["lang_ids"] = create_tensor(self.input_features, "lang_ids",
                                       torch.long, device)
    inputs["extra_args"] = {}
    return inputs

class DependencyParsingDataFlow(DataFlow):
  def __init__(self, config, task_name, tokenizers, label_mapping):
    super().__init__(config, task_name, tokenizers, label_mapping)
    self._inv_label_mapping = {v: k for k, v in label_mapping.items()}

  @property
  def example_class(self):
    return DependencyParsingExample

  @property
  def minibatch_class(self):
    return DependencyParsingMiniBatch

  def process_example(self, example: DependencyParsingExample):
    example.process(tokenizers=self.tokenizers,
                    label_mapping=self.label_mapping,
                    language_name2id=None)

  def convert_examples_to_features(self, examples: List[DependencyParsingExample]):
    examples: List[DependencyParsingExample]
    features = []

    max_token_length = max([example.len for example in examples])

    for idx, example in enumerate(examples):
      padding = [0] * (max_token_length - example.len)
      input_ids = example.input_ids + padding
      input_mask = example.input_mask + padding
      segment_ids = example.segment_ids + padding
      is_head = example.is_head + [2] * (max_token_length - example.len)

      if example.label_ids is not None and example.arcs_ids is not None:
        label_ids = example.label_ids + [example.label_padding_id] * (max_token_length - example.len)
        arcs_ids = example.arcs_ids + [-1] * (max_token_length - example.len)
      else:
        label_ids = None
        arcs_ids = None

      features.append(DependencyParsingFeature(
        input_ids=input_ids,
        input_mask=input_mask,
        is_head=is_head,
        segment_ids=segment_ids,
        label_ids=label_ids,
        arcs_ids=arcs_ids))

    return features


        