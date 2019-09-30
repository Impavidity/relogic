import json
from typing import List, Tuple, Dict
import os

import torch

from relogic.logickit.dataflow import DataFlow, Example, Feature, MiniBatch
from relogic.logickit.tokenizer.tokenization import BertTokenizer
from relogic.logickit.utils import create_tensor


class SequenceExample(Example):
  """SequenceExample contains the attributes and functionality of an Sequence Labeling example.

  Args:
    text (str): A sentence string.
    labels (List[str]): A list of labels.
  """
  def __init__(self, text, labels=None):
    super(SequenceExample, self).__init__()
    self.text = text
    self.raw_tokens = text.split()
    self.labels = labels
    self.label_padding = "X"

  def process(self, tokenizers: Dict, *inputs, **kwargs):
    """Process the Sequence Example..
    """
    for tokenizer in tokenizers.values():
      if isinstance(tokenizer, BertTokenizer):
        # BERT process part
        self.text_tokens, self.text_is_head = tokenizer.tokenize(self.text)
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

  @classmethod
  def from_structure(cls, structure):
    return cls(text=structure.text)

  @classmethod
  def from_json(cls, example):
    return cls(text=" ".join(example["tokens"]),
               labels=example.get("labels", None))

  @property
  def len(self):
    return len(self.input_ids)

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

    inputs["extra_args"] = {}
    return inputs

class SequenceDataFlow(DataFlow):
  def __init__(self, config, task_name, tokenizers, label_mapping):
    super(SequenceDataFlow, self).__init__(config, task_name, tokenizers, label_mapping)

  @property
  def example_class(self):
    return SequenceExample

  @property
  def minibatch_class(self):
    return SequenceMiniBatch

  def process_example(self, example: SequenceExample):
    example.process(tokenizers=self.tokenizers,
                    label_mapping=self.label_mapping)

  def convert_examples_to_features(self, examples: List[SequenceExample]):
    examples: List[SequenceExample]
    features = []

    max_token_length = max([example.len for example in examples])


    for idx, example in enumerate(examples):
      padding = [0] * (max_token_length - example.len)
      input_ids = example.input_ids + padding
      input_mask = example.input_mask + padding
      segment_ids = example.segment_ids + padding
      is_head = example.is_head + [2] * (max_token_length - example.len)

      if example.label_ids is not None:
        # We assume the label length is same as sequence length
        label_ids = example.label_ids + [example.label_padding_id] * (max_token_length - example.len)
      else:
        label_ids = None

      features.append(SequenceFeature(
        input_ids=input_ids,
        input_mask=input_mask,
        is_head=is_head,
        segment_ids=segment_ids,
        label_ids=label_ids))

    return features