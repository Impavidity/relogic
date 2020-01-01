from typing import List

import torch

from relogic.logickit.dataflow import DataFlow, Example, Feature, MiniBatch
from relogic.logickit.tokenizer.tokenization import BertTokenizer
from relogic.logickit.utils import create_tensor, truncate_seq_pair


class PointwiseExample(Example):
  """Pointwise Example contains the attributes and functionality of
  a pointwise example. This class is mainly used for sentence pair
  classification task, such as information retrieval, sentence similarity,
  with cross-entropy loss function.

  In this example, we call the first sentence as text_a, and the second sentence
  as text_b.

  Args:
    guid (str): global unique identifier for this pair. This information is usually used
      in the ranking problem.
    text (str):
  """
  def __init__(self, guid: str, text_a: str, text_b:str, label=None):
    super(PointwiseExample, self).__init__()
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label

  def process(self, tokenizers, *inputs, regression=False, **kwargs):
    """Process the sentence pair.

    :param tokenizers:
    :param inputs:
    :param kwargs:
    :return:
    """
    for tokenizer in tokenizers.values():
      if isinstance(tokenizer, BertTokenizer):
        self.text_a_tokens, self.text_a_is_head = tokenizer.tokenize(self.text_a)
        self.text_b_tokens, self.text_b_is_head = tokenizer.tokenize(self.text_b)

        max_seq_length = kwargs.pop("max_seq_length", 512)

        truncate_seq_pair(self.text_a_tokens, self.text_b_tokens, max_seq_length - 3)
        truncate_seq_pair(self.text_a_is_head, self.text_b_is_head, max_seq_length - 3)

        self.tokens = ["[CLS]"] + self.text_a_tokens + ["[SEP]"] + self.text_b_tokens + ["[SEP]"]
        self.segment_ids = [0] * (len(self.text_a_tokens) + 2) + [1] * (len(self.text_b_tokens) + 1)
        self.is_head = [2] + self.text_a_is_head + [2] + self.text_b_is_head + [2]

        self.input_ids = tokenizer.convert_tokens_to_ids(self.tokens)
        self.input_mask = [1] * len(self.input_ids)

        self.text_a_indices = []
        self.text_b_indices = []
        for idx, ind in enumerate(self.text_a_is_head):
          if ind == 1:
            self.text_a_indices.append(idx + 1)
        for idx, ind in enumerate(self.text_b_is_head):
          if ind == 1:
            self.text_b_indices.append(idx + len(self.text_a_is_head) + 2)

        if self.label is not None:
          if regression:
            # Regression Problem with sigmoid loss function
            self.label_ids = float(self.label)
          else:
            label_mapping = kwargs.get("label_mapping")
            self.label_ids = label_mapping[self.label]


  @classmethod
  def from_structure(cls, structure):
    return cls(guid="", text_a=structure.text_a, text_b=structure.text_b)

  @classmethod
  def from_json(cls, example):
    """
    """
    if isinstance(example["text_a"], str) and isinstance(example["text_b"], str):
      return cls(guid="{}|{}".format(example.get("text_a_id", 0), example.get("text_b_id", 0)),
               text_a=example["text_a"],
               text_b=example["text_b"],
               label=example.get("label", None))
    else:
      return cls(guid="{}|{}".format(example.get("text_a_id", 0), example.get("text_b_id", 0)),
                 text_a=" ".join(example["text_a"]),
                 text_b=" ".join(example["text_b"]),
                 label=example.get("label", None))

  @property
  def len(self):
      return len(self.input_ids)


class PointwiseFeature(Feature):
  """

  """
  def __init__(self, *inputs, **kwargs):
    super(PointwiseFeature, self).__init__()
    # BERT based feature
    self.input_ids = kwargs.pop("input_ids")
    self.input_mask = kwargs.pop("input_mask")
    self.is_head = kwargs.pop("is_head")
    self.segment_ids = kwargs.pop("segment_ids")
    self.text_a_indices = kwargs.pop("text_a_indices")
    self.text_b_indices = kwargs.pop("text_b_indices")
    self.label_ids = kwargs.pop("label_ids")

class PointwiseMiniBatch(MiniBatch):
  """

  """
  def __init__(self, *inputs, **kwargs):
    super(PointwiseMiniBatch, self).__init__(*inputs, **kwargs)

  def generate_input(self, device, use_label):
    """Generate tensors based on PointwiseFeatures
    """
    # BERT based feature
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
    inputs["text_a_indices"] = create_tensor(self.input_features, "text_a_indices",
                                             torch.long, device)
    inputs["text_b_indices"] = create_tensor(self.input_features, "text_b_indices",
                                             torch.long, device)
    if use_label:
      if self.config.regression:
        inputs["label_ids"] = create_tensor(self.input_features, "label_ids",
                                            torch.float, device)
      else:
        inputs["label_ids"] = create_tensor(self.input_features, "label_ids",
                                            torch.long, device)
    else:
      inputs["label_ids"] = None

    inputs["extra_args"] = {}
    if self.config.tasks[self.task_name]["selected_non_final_layers"] is not None:
      inputs["extra_args"]["selected_non_final_layers"] = self.config.tasks[self.task_name]["selected_non_final_layers"]

    return inputs

class PointwiseDataFlow(DataFlow):
  """DataFlow implementation based on Pointwise task"""
  def __init__(self, config, task_name, tokenizers, label_mapping):
    super(PointwiseDataFlow, self).__init__(config, task_name, tokenizers, label_mapping)

  @property
  def example_class(self):
    return PointwiseExample

  @property
  def minibatch_class(self):
    return PointwiseMiniBatch

  def process_example(self, example: PointwiseExample):
    """Process Pointwise example"""
    example.process(tokenizers=self.tokenizers,
                    label_mapping=self.label_mapping,
                    regression=self.config.regression)

  def convert_examples_to_features(self, examples: List[PointwiseExample]):
    examples: List[PointwiseExample]
    features = []

    # BERT based variables
    max_token_length = max([example.len for example in examples])
    max_text_a_indices_length = max([len(example.text_a_indices) for example in examples])
    max_text_b_indices_length = max([len(example.text_b_indices) for example in examples])
    for idx, example in enumerate(examples):
      # BERT based feature process
      padding = [0] * (max_token_length - example.len)
      input_ids = example.input_ids + padding
      input_mask = example.input_mask + padding
      segment_ids = example.segment_ids + padding
      is_head = example.is_head + padding
      a_indices_padding = [0] * (max_text_a_indices_length - len(example.text_a_indices))
      b_indices_padding = [0] * (max_text_b_indices_length - len(example.text_b_indices))
      text_a_indices = example.text_a_indices + a_indices_padding
      text_b_indices = example.text_b_indices + b_indices_padding
      if hasattr(example, "label_ids"):
        label_ids = example.label_ids
      else:
        label_ids = None

      features.append(
        PointwiseFeature(input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         is_head=is_head,
                         text_a_indices=text_a_indices,
                         text_b_indices=text_b_indices,
                         label_ids=label_ids))
    return features

  def decode_to_labels(self, preds, mbs: PointwiseMiniBatch):
    return preds