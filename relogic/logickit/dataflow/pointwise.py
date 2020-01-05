from typing import List

import torch

from relogic.logickit.dataflow import DataFlow, Example, Feature, MiniBatch
from relogic.logickit.tokenizer.tokenization import BertTokenizer
from relogic.logickit.utils import create_tensor, truncate_seq_pair

# A quick hard code

SEQUENCE_LABEL_MAPPING = {
  "_X_": 0,
  "I": 1,
  "O": 2
}

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
    text_a (str): Query tokens
    text_b (str): Candidate tokens
    sequence_labels (List[str]): This annotation is for evidence supporting in the matching task.
  """
  def __init__(self, guid: str, text_a: str, text_b:str, label=None,
               sequence_labels=None, selected_a_indices=None, selected_indices=None):
    super(PointwiseExample, self).__init__()
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label
    self.sequence_labels = sequence_labels
    self.selected_a_indices = selected_a_indices
    self.selected_indices = selected_indices
    self.sequence_label_padding = "_X_"

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

        if self.sequence_labels is not None:
          label_mapping = SEQUENCE_LABEL_MAPPING
          self.sequence_label_padding_id = label_mapping[self.sequence_label_padding]
          self.sequence_labels_ids = [self.sequence_label_padding_id] * len(self.input_ids)
          assert len(self.text_b_indices) == len(self.sequence_labels)
          for idx, label in zip(self.text_b_indices, self.sequence_labels):
            self.sequence_labels_ids[idx] = label_mapping[label]

        if self.selected_indices is not None:
          # Need to process span info
          self.text_b_full_token_spans = []
          offset = len(self.text_a_is_head) + 2
          start = -1
          for idx, ind in enumerate(self.text_b_is_head):
            if ind == 1:
              if start != -1:
                end = idx
                self.text_b_full_token_spans.append((offset + start, offset + end))
              start = idx
          self.text_b_full_token_spans.append((offset + start, offset + len(self.text_b_is_head)))
        if self.selected_a_indices is not None:
          self.text_a_full_token_spans = []
          offset = 1 # [CLS]
          start = -1
          for idx, ind in enumerate(self.text_a_is_head):
            if ind == 1:
              if start != -1:
                end = idx
                self.text_a_full_token_spans.append((offset + start, offset + end))
              start = idx
          self.text_a_full_token_spans.append((offset + start, offset + len(self.text_a_is_head)))


  @classmethod
  def from_structure(cls, structure):
    return cls(guid="",
               text_a=structure.text_a,
               text_b=structure.text_b,
               selected_a_indices=structure.selected_a_indices)

  @classmethod
  def from_json(cls, example):
    """
    """
    return cls(guid="{}|{}".format(example.get("text_a_id", 0), example.get("text_b_id", 0)),
               text_a=example["text_a"],
               text_b=example["text_b"],
               selected_indices=example.get("selected_indices", None),
               selected_a_indices=example.get("selected_a_indices", None),
               sequence_labels=example.get("sequence_labels", None),
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
    self.text_a_indices = kwargs.pop("text_a_indices", None)
    self.text_b_indices = kwargs.pop("text_b_indices", None)
    self.label_ids = kwargs.pop("label_ids", None)
    self.sequence_labels_ids = kwargs.pop("sequence_labels_ids", None)
    self.token_spans = kwargs.pop("token_spans", None)
    self.selected_indices = kwargs.pop("selected_indices", None)

    self.token_a_spans = kwargs.pop("token_a_spans", None)
    self.selected_a_indices = kwargs.pop("selected_a_indices", None)

class PointwiseMiniBatch(MiniBatch):
  """

  """
  def __init__(self, *inputs, **kwargs):
    super(PointwiseMiniBatch, self).__init__(*inputs, **kwargs)
    self.extra_features = kwargs.pop("extra_features", None)

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
    inputs["token_spans"] = create_tensor(self.input_features, "token_spans",
                                          torch.long, device)
    inputs["selected_indices"] = create_tensor(self.input_features, "selected_indices",
                                               torch.long, device)
    inputs["token_a_spans"] = create_tensor(self.input_features, "token_a_spans",
                                            torch.long, device)
    inputs["selected_a_indices"] = create_tensor(self.input_features, "selected_a_indices",
                                                 torch.long, device)
    if use_label:
      if self.config.regression:
        inputs["label_ids"] = create_tensor(self.input_features, "label_ids",
                                            torch.float, device)
      else:
        inputs["label_ids"] = create_tensor(self.input_features, "label_ids",
                                            torch.long, device)
      if hasattr(self.config, "doc_ir_model") and self.config.doc_ir_model == "evidence":
        inputs["label_ids"] = (inputs["label_ids"], create_tensor(
          self.input_features, "sequence_labels_ids", torch.long, device))
    else:
      inputs["label_ids"] = None

    inputs["extra_args"] = {}
    if self.config.tasks[self.task_name]["selected_non_final_layers"] is not None:
      inputs["extra_args"]["selected_non_final_layers"] = self.config.tasks[self.task_name]["selected_non_final_layers"]
    if self.extra_features is not None:
      # we currently assume it is document level label and edge data
      if use_label:
        inputs["label_ids"] = create_tensor(self.extra_features, "label_ids",
                                            torch.long, device)
      inputs["extra_args"]["edge_data"] = [feature.edge_data for feature in self.extra_features]
      inputs["extra_args"]["doc_span"] = create_tensor(self.extra_features, "doc_span",
                                                       torch.long, device)
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
    try:
      max_a_full_token_length = max([len(example.text_a_full_token_spans) for example in examples])
      max_selected_a_indices_length = max([len(example.selected_a_indices) for example in examples])
    except:
      max_a_full_token_length = 0
      max_selected_a_indices_length = 0
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
      if example.selected_a_indices is not None:
        # Need to do the selection, so the token span and indices are needed.
        token_a_spans = example.text_a_full_token_spans + [(1, 0)] * (
              max_a_full_token_length - len(example.text_a_full_token_spans))
        selected_a_indices = example.selected_a_indices + [-1] * (
              max_selected_a_indices_length - len(example.selected_a_indices))
      else:
        token_a_spans = None
        selected_a_indices = None
      if hasattr(example, "label_ids"):
        label_ids = example.label_ids
      else:
        label_ids = None

      if hasattr(self.config, "doc_ir_model") and self.config.doc_ir_model == "evidence":
        if hasattr(example, "sequence_labels_ids"):
          sequence_labels_ids = example.sequence_labels_ids + padding
        else:
          sequence_labels_ids = [0] * example.len + padding
      else:
        sequence_label_ids = None

      features.append(
        PointwiseFeature(input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         is_head=is_head,
                         text_a_indices=text_a_indices,
                         text_b_indices=text_b_indices,
                         token_a_spans=token_a_spans,
                         selected_a_indices=selected_a_indices,
                         label_ids=label_ids,
                         sequence_labels_ids=sequence_labels_ids))
    return features

  def decode_to_labels(self, preds, mbs: PointwiseMiniBatch):
    return preds