"""
The module contains the implementation of DataFlow on SRL task.
"""

from typing import List, Tuple

import torch

from relogic.logickit.base.constants import SRL_LABEL_SEQ_BASED, SRL_LABEL_SPAN_BASED
from relogic.logickit.dataflow import DataFlow, Example, Feature, MiniBatch
from relogic.logickit.tokenizer.tokenization import BertTokenizer
from relogic.logickit.utils import create_tensor
from relogic.structures import enumerate_spans


class SRLExample(Example):
  """SRLExample contains the attributes and functionality of an SRL example.

  Args:
    text (str): A sentence string.
    labels (List[Tuple]): A list of labels. Each label is a Tuple containing
      predicate_start_index, predicate_end_index, predicate_text_string,
      argument_start_index, argument_end_index, argument_text_string, argument_label.

  """
  def __init__(self, text, labels=None):
    super(SRLExample, self).__init__()
    self.text = text
    self.raw_tokens = text.split()
    self.labels = labels
    # Hard code here
    self.label_padding = 'X'

  def process(self, tokenizer, *inputs, **kwargs):
    """Process the SRL example.

    This process requires the tokenizer. Furthermore, if this example is for
     training and evaluation, the label_format and label_mapping are required.
    """

    if isinstance(tokenizer, BertTokenizer):
      # BERT process part
      self.text_tokens, self.text_is_head = tokenizer.tokenize(self.text)
      self.tokens = ["[CLS]"] + self.text_tokens + ["[SEP]"]
      self.segment_ids = [0] * (len(self.text_tokens) + 2)
      self.is_head = [2] + self.text_is_head + [2]
      self.head_index = [
          idx for idx, value in enumerate(self.is_head) if value == 1
      ] + [len(self.is_head) - 1]
      # index of [SEP] is len(self.is_head) - 1.
      # Assume we have sentence [A, B, C]. After processing,
      # it becomes [[CLS], A, B, C, [SEP]]
      # length of head_index is 3, that is [1, 2, 3].
      # We pad it with len(self.is_head) - 1, which is 4.
      # Assume we have span (2, 3) = C, exclusive.
      # self.head_index[2] =  3, self.head_index[3] = 4.
      # So the span for tokenized sentence is (3, 4) = C

      self.input_ids = tokenizer.convert_tokens_to_ids(self.tokens)
      self.input_mask = [1] * len(self.input_ids)

      # Enumerate the span and map the spans into sub-token level
      spans = enumerate_spans(sentence=self.raw_tokens, max_span_width=20)

      self.enumerated_span_candidates: List[Tuple[int, int]] = []
      for span in spans:
        self.enumerated_span_candidates.append(
            (self.head_index[span[0]], self.head_index[span[1]]))

      # Currently we assume that the predicate length is 1
      spans = enumerate_spans(sentence=self.raw_tokens, max_span_width=1)
      self.predicate_candidates: List[Tuple[int, int]] = []
      for span in spans:
        self.predicate_candidates.append(
            (self.head_index[span[0]], self.head_index[span[1]]))

      # If this is for model development, then self.labels is not None
      # If this is for deployment, then self.labels is None
      if self.labels is not None:
        label_mapping = kwargs.pop("label_mapping")
        self.label_padding_id = label_mapping[self.label_padding]
        label_format = kwargs.pop("label_format")

        if label_format == SRL_LABEL_SPAN_BASED:
          self.label_ids = []

          for label in self.labels:
            (predicate_start, predicate_end, predicate_text,
             arg_start, arg_end, argument_text, arg_label) = label

            label_tuple = (self.head_index[predicate_start],
                           self.head_index[predicate_end],
                           self.head_index[arg_start],
                           self.head_index[arg_end], label_mapping[arg_label])
            self.label_ids.append(label_tuple)

        elif label_format == SRL_LABEL_SEQ_BASED:
          pass
        else:
          raise ValueError()
    else:
      # Traditional process part
      pass

  @classmethod
  def from_structure(cls, structure):
    """Implementation of converting structure into SRLExample."""

    return cls(text=structure.text)

  @classmethod
  def from_json(cls, example):
    """Implementation of converting json object into SRLExample.

    This object can be grouped all predicates together or represents a single predicate.

    """
    return cls(text=" ".join(example["tokens"]),
               labels=example.get("labels", None))

  @property
  def len(self):
    """SRL example tokens length (after all preprocessing)."""

    return len(self.tokens)

  @property
  def label_len(self):
    """SRL example label length (after all preprocessing). Can be span-based or sequence-based"""

    if hasattr(self, "label_ids"):
      return len(self.label_ids)
    return 0


class SRLFeature(Feature):
  """SRL features.
  Some extra features such as arg_candidates and predicate_candidates

  """
  def __init__(self, *inputs, **kwargs):
    super(SRLFeature, self).__init__()
    self.input_ids = kwargs.pop("input_ids")
    self.input_mask = kwargs.pop("input_mask")
    self.segment_ids = kwargs.pop("segment_ids")
    self.is_head = kwargs.pop("is_head")
    self.label_ids = kwargs.pop("label_ids")
    self.arg_candidates = kwargs.pop("arg_candidates")
    self.predicate_candidates = kwargs.pop("predicate_candidates")


class SRLMiniBatch(MiniBatch):
  """MiniBatch Implementation based on SRL task.

  """
  def __init__(self, *inputs, **kwargs):
    super(SRLMiniBatch, self).__init__(*inputs, **kwargs)

  def generate_input(self, device, use_label):
    """Generate tensors based on SRL Features."""

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
      inputs["label_ids"] = create_tensor(self.input_features, "label_ids",
                                          torch.long, device)
    else:
      inputs["label_ids"] = None
    inputs["arg_candidates"] = create_tensor(
        self.input_features, "arg_candidates", torch.long, device)
    inputs["predicate_candidates"] = create_tensor(
        self.input_features, "predicate_candidates", torch.long, device)
    return inputs


class SRLDataFlow(DataFlow):
  """DataFlow implementation based on SRL task."""
  def __init__(self, config, task_name, tokenizer, label_mapping):
    super(SRLDataFlow, self).__init__(config, task_name, tokenizer, label_mapping)

  @property
  def example_class(self):
    """Return SRLExample class"""

    return SRLExample

  @property
  def minibatch_class(self):
    """Return SRLMiniBatch class"""

    return SRLMiniBatch

  def process_example(self, example: SRLExample):
    """Process SRL example with extra arguments.
    Including label_mapping and srl_label_format

    """
    example.process(tokenizer=self.tokenizer,
                    label_mapping=self.label_mapping,
                    label_format=self.config.srl_label_format)

  def convert_examples_to_features(self, examples: List[SRLExample]):
    """
    TODO: The implementation is based on BERT model.
    The convert for other models will be implemeted.

    """
    examples: List[SRLExample]
    label_format = self.config.srl_label_format
    features = []
    max_token_length = max([example.len for example in examples])
    if label_format is not None:
      max_label_length = max([example.label_len for example in examples])
    else:
      max_label_length = None

    max_arg_candidate_length = max(
        [len(example.enumerated_span_candidates) for example in examples])
    max_predicate_candidate_length = max(
        [len(example.predicate_candidates) for example in examples])

    for idx, example in enumerate(examples):
      padding = [0] * (max_token_length - example.len)
      input_ids = example.input_ids + padding
      input_mask = example.input_mask + padding
      segment_ids = example.segment_ids + padding
      is_head = example.is_head + [2] * (max_token_length - example.len)

      arg_candidates = example.enumerated_span_candidates + [
          (1, 0)
      ] * (max_arg_candidate_length - len(example.enumerated_span_candidates))
      predicate_candidates = example.predicate_candidates + [
          (1, 0)
      ] * (max_predicate_candidate_length - len(example.predicate_candidates))

      # For label processing, there are three choice
      # 1. None if it is for prediction
      # 2. Sequence Labeling format, then it is just List[str]. This is predicate based.
      # 3. Span format, then it is List[Tuple[int, int, int, int, int]]
      if label_format is None:
        label_ids = None
      elif label_format == SRL_LABEL_SPAN_BASED:
        # span (1, 0) is a invalid span. We use that as padding
        label_ids = example.label_ids + [
            (1, 0, 1, 0, example.label_padding_id)
        ] * (max_label_length - example.label_len)
      elif label_format == SRL_LABEL_SEQ_BASED:
        label_ids = example.label_ids + [example.label_padding_id] * (
            max_label_length - example.label_len)
      else:
        raise ValueError(
            "The label format {} is not supported. Choice: {} | {}".format(
                label_format, SRL_LABEL_SEQ_BASED, SRL_LABEL_SPAN_BASED))

      features.append(
          SRLFeature(input_ids=input_ids,
                     input_mask=input_mask,
                     segment_ids=segment_ids,
                     is_head=is_head,
                     label_ids=label_ids,
                     arg_candidates=arg_candidates,
                     predicate_candidates=predicate_candidates))
    return features
