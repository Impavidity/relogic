import json
from typing import List, Tuple, Dict
import os

import torch

from relogic.logickit.dataflow import DataFlow, Example, Feature, MiniBatch
from relogic.logickit.tokenizer.tokenization import BertTokenizer
from relogic.logickit.utils import create_tensor, filter_head_prediction


class SingletonExample(Example):
  """SingletonExample contains the attributes and functionality of an Sentence Classification example.

  Args:
    text (str): A sentence string.
    labels (str or List[str]): A label string or a list of label string.
  """
  def __init__(self, text, labels=None):
    super(SingletonExample, self).__init__()
    self.text = text
    self.raw_tokens = text.split()
    self.labels = labels

  def process(self, tokenizers: Dict, *inputs, **kwargs):
    """Process the Singleton Example..
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
          if isinstance(self.labels, str):
            # Single Label Classfication
            self.label_ids = label_mapping[self.labels]
          elif isinstance(self.labels, list):
            # Multi Label Classification
            label_size = len(label_mapping)
            self.label_ids = [0] * label_size
            for label in self.labels:
              self.label_ids[label_mapping[label]] = 1
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

class SingletonFeature(Feature):
  """Singleton Features.
  """
  def __init__(self, *inputs, **kwargs):
    super(SingletonFeature, self).__init__()
    self.input_ids = kwargs.pop("input_ids")
    self.input_mask = kwargs.pop("input_mask")
    self.segment_ids = kwargs.pop("segment_ids")
    self.is_head = kwargs.pop("is_head")
    self.label_ids = kwargs.pop("label_ids")

class SingletonMiniBatch(MiniBatch):
  """

  """
  def __init__(self, *inputs, **kwargs):
    super(SingletonMiniBatch, self).__init__(*inputs, **kwargs)

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

    # inputs["extra_args"] = {
    #   "selected_non_final_layers": [10]
    # }
    inputs["extra_args"] = {}

    if self.config.tasks[self.task_name]["selected_non_final_layers"] is not None:
      inputs["extra_args"]["selected_non_final_layers"] = self.config.tasks[self.task_name]["selected_non_final_layers"]


    return inputs

class SingletonDataFlow(DataFlow):
  def __init__(self, config, task_name, tokenizers, label_mapping):
    super(SingletonDataFlow, self).__init__(config, task_name, tokenizers, label_mapping)
    self._inv_label_mapping = {v: k for k, v in label_mapping.items()}

  @property
  def example_class(self):
    return SingletonExample

  @property
  def minibatch_class(self):
    return SingletonMiniBatch

  def process_example(self, example: SingletonExample):
    example.process(tokenizers=self.tokenizers,
                    label_mapping=self.label_mapping)

  def convert_examples_to_features(self, examples: List[SingletonExample]):
    examples: List[SingletonExample]
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
        label_ids = example.label_ids
      else:
        label_ids = None

      features.append(SingletonFeature(
        input_ids=input_ids,
        input_mask=input_mask,
        is_head=is_head,
        segment_ids=segment_ids,
        label_ids=label_ids))

    return features

  def decode_to_labels(self, preds, mb: MiniBatch):
    labels = []
    for example, pred_logits in zip(mb.examples, preds):
      pred_probs = torch.sigmoid(pred_logits).data.cpu().numpy()
      pred_label = []
      for idx, pred_prob in enumerate(pred_probs):
        if pred_prob > 0.5:
          pred_label.append(self._inv_label_mapping[idx])
      labels.append(pred_label)
    return labels



