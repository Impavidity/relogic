"""
The module contains the implementation of DataFlow on SRL task.
"""

import json
from typing import List, Tuple
import os

import torch

from relogic.logickit.base.constants import SRL_LABEL_SEQ_BASED, SRL_LABEL_SPAN_BASED, SRL_PREDICATE_EXTRA_SURFACE
from relogic.logickit.dataflow import DataFlow, Example, Feature, MiniBatch
from relogic.logickit.tokenizer.fasttext_tokenization import FasttextTokenizer
from relogic.logickit.utils import create_tensor, get_span_labels
from relogic.structures import enumerate_spans
from transformers.tokenization_utils import PreTrainedTokenizer

class SRLExample(Example):
  """SRLExample contains the attributes and functionality of an SRL example.

  Args:
    text (str): A sentence string.
    labels (List[Tuple] or List[Str]): A list of labels. Each label is a Tuple containing
      predicate_start_index, predicate_end_index, predicate_text_string,
      argument_start_index, argument_end_index, argument_text_string, argument_label.

      If it is sequence labeling, then the labels are a sequence of labels string.

  """
  def __init__(self,
               text,
               predicate_text=None,
               predicate_index=None,
               labels=None,
               pos_tag_label=None,
               predicate_sense_label=None,
               label_candidates=None):
    super(SRLExample, self).__init__()
    self.text = text
    self.raw_tokens = text.split(" ")
    self.labels = labels
    # Hard code here
    self.label_padding = 'X'
    self.predicate_text = predicate_text
    if self.predicate_text is not None:
      self.raw_predicate_text_tokens = self.predicate_text.split()
    self.predicate_index = predicate_index
    # We assume the index has only one value

    self.pos_tag_label = pos_tag_label
    self.predicate_sense_label = predicate_sense_label
    self.label_candidates = label_candidates

  def process(self, tokenizers, *inputs, **kwargs):
    """Process the SRL example.

    This process requires the tokenizer. Furthermore, if this example is for
     training and evaluation, the label_format and label_mapping are required.

    History Log: Originally we use adapted tokenizer for determining the token head
      for the pretokenized sentence. Here we plan to change it with for loop.
      Then we do not need to adapt it for each tokenizer.
      ```
      self.text_tokens, self.text_is_head = tokenizer.tokenize(self.text)
      ```
      We will adapt the code from transformer example run_ner.py and utils_ner.py
    """
    use_gold_predicate = kwargs.pop("use_gold_predicate", False)
    use_gold_argument = kwargs.pop("use_gold_argument", False)
    predicate_reveal_method = kwargs.pop("predicate_reveal_method", SRL_PREDICATE_EXTRA_SURFACE)
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
        # Currently it only support one group of model (BERT).
        self.tokens = [tokenizer.cls_token] + self.text_tokens + [tokenizer.sep_token]
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
        assert (len(self.tokens) == len(self.is_head))

        # This part is mainly for the arxiv version paper.
        if predicate_reveal_method == SRL_PREDICATE_EXTRA_SURFACE:
          # In this case we assume the input only has one predicate
          # User needs to separate them by themselves.

          assert self.predicate_text is not None
          predicate_text_tokens = []
          for word in self.raw_predicate_text_tokens:
            word_tokens = tokenizer.tokenize(word)
            predicate_text_tokens.extend(word_tokens)
            # self.text_is_head.extend([1] + [0] * (len(word_tokens) - 1))
            # We ignore the is_head information for predicate for now
          predicate_text_tokens += [tokenizer.sep_token]
          self.tokens.extend(predicate_text_tokens)
          self.segment_ids += [1] * len(predicate_text_tokens)
          assert self.predicate_index is not None
          self.predicate_idx = self.head_index[self.predicate_index]


        self.input_ids = tokenizer.convert_tokens_to_ids(self.tokens)
        self.input_mask = [1] * len(self.input_ids)

        # Enumerate the span and map the spans into sub-token level
        if use_gold_argument:
          assert self.labels is not None
          spans = []
          for label in self.labels:
            (predicate_start, predicate_end, predicate_text,
             arg_start, arg_end, argument_text, arg_label) = label
            span_tuple = (arg_start, arg_end)
            if span_tuple not in spans:
              spans.append(span_tuple)
        else:
          spans = enumerate_spans(sentence=self.raw_tokens, max_span_width=30)

        self.enumerated_span_candidates: List[Tuple[int, int]] = []
        for span in spans:
          self.enumerated_span_candidates.append(
              (self.head_index[span[0]], self.head_index[span[1]]))

        # Currently we assume that the predicate length is 1
        if use_gold_predicate:
          assert self.labels is not None
          spans = []
          for label in self.labels:
            (predicate_start, predicate_end, predicate_text,
             arg_start, arg_end, argument_text, arg_label) = label
            span_tuple = (predicate_start, predicate_end)
            if span_tuple not in spans:
              spans.append(span_tuple)
        else:
          spans = enumerate_spans(sentence=self.raw_tokens, max_span_width=1)

        self.predicate_candidates: List[Tuple[int, int]] = []
        for span in spans:
          self.predicate_candidates.append(
              (self.head_index[span[0]], self.head_index[span[1]]))

        # If this is for model development, then self.labels is not None
        # If this is for deployment, then self.labels is None
        if self.labels is not None:
          label_mapping = kwargs.get("label_mapping")
          self.label_padding_id = label_mapping[self.label_padding]
          label_format = kwargs.get("label_format")

          if label_format == SRL_LABEL_SPAN_BASED:
            self.label_ids = []
            self.pred_span_label_dict = dict([(item, 0)
                                              for item in self.predicate_candidates])
            self.arg_span_label_dict = dict([(item, 0)
                                             for item in self.enumerated_span_candidates])

            for label in self.labels:
              (predicate_start, predicate_end, predicate_text,
               arg_start, arg_end, argument_text, arg_label) = label

              label_tuple = (self.head_index[predicate_start],
                             self.head_index[predicate_end],
                             self.head_index[arg_start],
                             self.head_index[arg_end], label_mapping[arg_label])
              self.label_ids.append(label_tuple)
              self.pred_span_label_dict[(self.head_index[predicate_start], self.head_index[predicate_end])] = 1
              self.arg_span_label_dict[(self.head_index[arg_start], self.head_index[arg_end])] = 1

            self.pred_span_label_ids = [self.pred_span_label_dict[item] for item in self.predicate_candidates]
            self.arg_span_label_ids = [self.arg_span_label_dict[item] for item in self.enumerated_span_candidates]

          elif label_format == SRL_LABEL_SEQ_BASED:
            label_mapping = kwargs.get("label_mapping")
            self.label_padding_id = label_mapping[self.label_padding]
            self.label_ids = [self.label_padding_id] * len(self.tokens)
            assert len(self.labels) == len(self.head_index[:-1])
            for idx, label in zip(self.head_index[:-1], self.labels):
              # we ignore the index of [SEP]
              self.label_ids[idx] = label_mapping[label]
          else:
            raise ValueError()

        # Process the pos tag labels into sequence labeling task
        if self.pos_tag_label is not None:
          pos_tag_label_mapping = kwargs.get("pos_tag_label_mapping")
          self.pos_tag_label_padding_id = pos_tag_label_mapping[self.label_padding]
          self.pos_tag_label_ids = [self.label_padding_id] * len(self.input_ids)
          for idx, label in zip(self.head_index, self.pos_tag_label):
            self.pos_tag_label_ids[idx] = pos_tag_label_mapping[label]

        if self.predicate_sense_label is not None:
          predicate_sense_mapping = kwargs.get("predicate_sense_mapping")
          self.predicate_sense_label_padding_id = predicate_sense_mapping[self.label_padding]
          self.predicate_sense_label_ids = [self.predicate_sense_label_padding_id] * len(self.input_ids)


      elif isinstance(tokenizer, FasttextTokenizer):
        # Traditional process part
        self._text_tokens = tokenizer.tokenize(self.text)
        # self._text_tokens, self._char_tokens = tokenizer.tokenize(self.text)
        self._input_token_ids = tokenizer.convert_tokens_to_ids(self._text_tokens)
        # self._input_char_ids = tokenizer.convert_char_to_ids(self._char_tokens)

        spans = enumerate_spans(sentence=self._text_tokens, max_span_width=20)

        self._enumerated_span_candidates: List[Tuple[int, int]] = []
        for span in spans:
          self._enumerated_span_candidates.append(span)

        spans = enumerate_spans(sentence=self._text_tokens, max_span_width=1)
        self._predicate_candidates: List[Tuple[int, int]] = []
        for span in spans:
          self._predicate_candidates.append(span)

        if self.labels is not None:
          label_mapping = kwargs.get("label_mapping")
          self.label_padding_id = label_mapping[self.label_padding]
          label_format = kwargs.get("label_format")

          if label_format == SRL_LABEL_SPAN_BASED:
            self._label_ids = []
            self._pred_span_label_dict = dict([(item, 0)
                                               for item in self._predicate_candidates])
            self._arg_span_label_dict = dict([(item, 0)
                                              for item in self._enumerated_span_candidates])
            for label in self.labels:
              (predicate_start, predicate_end, predicate_text,
               arg_start, arg_end, argument_text, arg_label) = label

              label_tuple = (predicate_start, predicate_end,
                             arg_start, arg_end, label_mapping[arg_label])
              self._label_ids.append(label_tuple)
              self._pred_span_label_dict[(predicate_start, predicate_end)] = 1
              self._arg_span_label_dict[(arg_start, arg_end)] = 1

            self._pred_span_label_ids = [self._pred_span_label_dict[item] for item in self._predicate_candidates]
            self._arg_span_label_ids = [self._arg_span_label_dict[item] for item in self._enumerated_span_candidates]
          elif label_format == SRL_LABEL_SEQ_BASED:
            pass
          else:
            raise ValueError

  @classmethod
  def from_structure(cls, structure):
    """Implementation of converting structure into SRLExample."""

    return cls(text=structure.tokenized_text,
               predicate_text=structure.predicate_text,
               predicate_index=structure.predicate_index)

  @classmethod
  def from_json(cls, example):
    """Implementation of converting json object into SRLExample.

    This object can be grouped all predicates together or represents a single predicate.

    """

    return cls(text=" ".join(example["tokens"]),
               labels=example.get("labels", None),
               pos_tag_label=example.get("pos_tag", None),
               predicate_sense_label=example.get("predicate_sense", None),
               label_candidates=example.get("label_candidates", None),
               predicate_text=example.get("predicate_text", None),
               predicate_index=example.get("predicate_index", None))

  @property
  def len(self):
    """SRL example tokens length (after all preprocessing)."""

    return len(self.tokens)

  @property
  def _len(self):
    """SRL example classical token length"""

    return len(self._input_token_ids)

  @property
  def label_len(self):
    """SRL example label length (after all preprocessing). Can be span-based or sequence-based"""

    if hasattr(self, "label_ids"):
      return len(self.label_ids)
    return 0

  @property
  def _label_len(self):
    """SRL example classical label length."""
    if hasattr(self, "_label_ids"):
      return len(self._label_ids)
    return 0


class SRLFeature(Feature):
  """SRL features.
  Some extra features such as arg_candidates and predicate_candidates

  """
  def __init__(self, *inputs, **kwargs):
    super(SRLFeature, self).__init__()
    # Bert based feature
    self.input_ids = kwargs.pop("input_ids")
    self.input_mask = kwargs.pop("input_mask")
    self.segment_ids = kwargs.pop("segment_ids")
    self.is_head = kwargs.pop("is_head")
    self.label_ids = kwargs.pop("label_ids")
    self.arg_candidates = kwargs.pop("arg_candidates")
    self.predicate_candidates = kwargs.pop("predicate_candidates")
    self.arg_candidate_label_ids = kwargs.pop("arg_candidate_label_ids")
    self.predicate_candidate_label_ids = kwargs.pop("predicate_candidate_label_ids")
    self.predicate_idx = kwargs.pop("predicate_idx")

    # Label embeddings
    self.label_candidates = kwargs.pop("label_candidates")
    self.label_candidates_mask = kwargs.pop("label_candidates_mask")

    # POS tag labels
    self.pos_tag_label_ids = kwargs.pop("pos_tag_label_ids")

    # Classical feature
    self._input_token_ids = kwargs.pop("_input_token_ids")
    self._label_ids = kwargs.pop("_label_ids")
    self._arg_candidates = kwargs.pop("_arg_candidates")
    self._predicate_candidates = kwargs.pop("_predicate_candidates")


class SRLMiniBatch(MiniBatch):
  """MiniBatch Implementation based on SRL task.

  """
  def __init__(self, *inputs, **kwargs):
    super(SRLMiniBatch, self).__init__(*inputs, **kwargs)

  def generate_input(self, device, use_label):
    """Generate tensors based on SRL Features."""
    # BERT based features
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
    inputs["predicate_idx"] = create_tensor(self.input_features, "predicate_idx",
                                            torch.long, device)
    if use_label:
      label_ids = create_tensor(self.input_features, "label_ids",
                                          torch.long, device)
      predicate_candidate_label_ids = create_tensor(self.input_features, "predicate_candidate_label_ids",
                                           torch.long, device)
      arg_candidate_label_ids = create_tensor(self.input_features, "arg_candidate_label_ids",
                                           torch.long, device)
      pos_tag_label_ids = create_tensor(self.input_features, "pos_tag_label_ids", torch.long, device)
      if predicate_candidate_label_ids is None or arg_candidate_label_ids is None:
        inputs["label_ids"] = label_ids
      else:
        inputs["label_ids"] = (label_ids, predicate_candidate_label_ids, arg_candidate_label_ids, pos_tag_label_ids)
    else:
      inputs["label_ids"] = None
    inputs["arg_candidates"] = create_tensor(
        self.input_features, "arg_candidates", torch.long, device)
    inputs["predicate_candidates"] = create_tensor(
        self.input_features, "predicate_candidates", torch.long, device)

    # Label Embeddings
    inputs["label_candidates"] = create_tensor(self.input_features, "label_candidates", 
        torch.long, device)
    inputs["label_candidates_mask"] = create_tensor(self.input_features, "label_candidates_mask", 
        torch.long, device)
    # Classical features
    # inputs["_input_token_ids"] = create_tensor(self.input_features, "_input_token_ids",
    #                                            torch.long, device)
    # inputs["_token_length"] = create_tensor(self.input_features, "_token_length",
    #                                         torch.long, device)
    # if use_label:
    #   _label_ids = create_tensor(self.input_features, "_label_ids",
    #                                       torch.long, device)
    #   _predicate_candidate_label_ids = create_tensor(self.input_features, "_predicate_candidate_label_ids",
    #                                        torch.long, device)
    #   _arg_candidate_label_ids = create_tensor(self.input_features, "_arg_candidate_label_ids",
    #                                        torch.long, device)
    #   if _predicate_candidate_label_ids is None or _arg_candidate_label_ids is None:
    #     inputs["_label_ids"] = _label_ids
    #   else:
    #     inputs["_label_ids"] = (_label_ids, _predicate_candidate_label_ids, _arg_candidate_label_ids)
    # else:
    #   inputs["_label_ids"] = None
    # inputs["_arg_candidates"] = create_tensor(
    #     self.input_features, "_arg_candidates", torch.long, device)
    # inputs["_predicate_candidates"] = create_tensor(
    #   self.input_features, "_predicate_candidates", torch.long, device)
    # inputs["extra_args"] = {
    #   "selected_non_final_layers": [1, 9, 18]
    # }

    return inputs


class SRLDataFlow(DataFlow):
  """DataFlow implementation based on SRL task."""
  def __init__(self, config, task_name, tokenizers, label_mapping):
    super(SRLDataFlow, self).__init__(config, task_name, tokenizers, label_mapping)
    self._inv_label_mapping = {v: k for k, v in label_mapping.items()}
    self.pos_tag_label_mapping = json.load(open("data/preprocessed_data/pos_tag.json"))
    self.predicate_sense_label_mapping = json.load(open("data/preprocessed_data/predicate_sense_label_mapping.json"))
    self.use_gold_predicate = hasattr(config, "srl_use_gold_predicate") and config.srl_use_gold_predicate
    self.use_gold_argument = hasattr(config, "srl_use_gold_argument") and config.srl_use_gold_argument
    if hasattr(config, "predicate_reveal_method"):
      self.predicate_reveal_method = config.predicate_reveal_method
    else:
      self.predicate_reveal_method = None

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
    example.process(tokenizers=self.tokenizers,
                    label_mapping=self.label_mapping,
                    pos_tag_label_mapping=self.pos_tag_label_mapping,
                    predicate_sense_mapping=self.predicate_sense_label_mapping,
                    label_format=self.config.srl_label_format,
                    use_gold_predicate=self.use_gold_predicate,
                    use_gold_argument=self.use_gold_argument,
                    predicate_reveal_method=self.predicate_reveal_method)

  def convert_examples_to_features(self, examples: List[SRLExample]):
    """
    TODO: The implementation is based on BERT model.
    The convert for other models will be implemeted.

    """
    examples: List[SRLExample]
    label_format = self.config.srl_label_format
    features = []

    # BERT based variables
    max_token_length = max([example.len for example in examples])
    if label_format is not None:
      max_label_length = max([example.label_len for example in examples])
    else:
      max_label_length = None

    max_arg_candidate_length = max(
        [len(example.enumerated_span_candidates) for example in examples])
    max_predicate_candidate_length = max(
        [len(example.predicate_candidates) for example in examples])

    # Label embedding variables
    try:
      max_label_candidate_length = max([
          max([len(label_candidates) for label_candidates in example.label_candidates])
        for example in examples])
    except:
      max_label_candidate_length = None

    # Classical based variable
    # _max_token_length = max([example._len for example in examples])
    # if label_format is not None:
    #   _max_label_length = max([example._label_len for example in examples])
    # else:
    #   _max_label_length = None
    #
    # _max_arg_candidate_length = max([
    #   len(example._enumerated_span_candidates) for example in examples])
    # _max_predicate_candidate_length = max(
    #   [len(example._predicate_candidates) for example in examples])

    for idx, example in enumerate(examples):
      # Bert based feature process
      padding = [0] * (max_token_length - example.len)
      input_ids = example.input_ids + padding
      input_mask = example.input_mask + padding
      segment_ids = example.segment_ids + padding
      is_head = example.is_head + [2] * (max_token_length - len(example.is_head))
      # Because we do not have is_head for extra predicate part, so we need to recalculate the
      # padding for is_head

      arg_candidates = example.enumerated_span_candidates + [
          (1, 0)
      ] * (max_arg_candidate_length - len(example.enumerated_span_candidates))
      predicate_candidates = example.predicate_candidates + [
          (1, 0)
      ] * (max_predicate_candidate_length - len(example.predicate_candidates))

      if example.predicate_index is not None:
        predicate_idx = example.predicate_idx
      else:
        predicate_idx = None

      # For label processing, there are three choice
      # 1. None if it is for prediction
      # 2. Sequence Labeling format, then it is just List[str]. This is predicate based.
      # 3. Span format, then it is List[Tuple[int, int, int, int, int]]
      if label_format is None or self.config.mode in ["deployment"]:
        label_ids = None
        arg_candidate_label_ids = None
        predicate_candidate_label_ids = None
      elif label_format == SRL_LABEL_SPAN_BASED:
        # span (1, 0) is a invalid span. We use that as padding
        label_ids = example.label_ids + [
            (1, 0, 1, 0, example.label_padding_id)
        ] * (max_label_length - example.label_len)

        arg_candidate_label_ids = example.arg_span_label_ids + [
          0] * (max_arg_candidate_length - len(example.enumerated_span_candidates))
        predicate_candidate_label_ids = example.pred_span_label_ids + [
          0] * (max_predicate_candidate_length - len(example.predicate_candidates))


      elif label_format == SRL_LABEL_SEQ_BASED:
        label_ids = example.label_ids + [example.label_padding_id] * (
            max_label_length - example.label_len)
        arg_candidate_label_ids = None
        predicate_candidate_label_ids = None
      else:
        raise ValueError(
            "The label format {} is not supported. Choice: {} | {}".format(
                label_format, SRL_LABEL_SEQ_BASED, SRL_LABEL_SPAN_BASED))

      # POS tag joint learning

      if example.pos_tag_label is not None:
        pos_tag_label_ids = example.pos_tag_label_ids + [example.pos_tag_label_padding_id] * (max_token_length - example.len)
      else:
        pos_tag_label_ids = None

      # Label embeddings
      # It will be a matrix. The first dim will be the length of the sentence (max), the second dim will
      # be the length of the label candidates (max)
      if max_label_candidate_length is not None:
        label_candidates_list = []
        label_candidates_mask_list = []
        
        for label_candidates in example.label_candidates:
          padded_label_candidates = label_candidates + [0] * (max_label_candidate_length - len(label_candidates))
          padded_label_candidates_mask = [1] * len(label_candidates) + [0] * (max_label_candidate_length - len(label_candidates))
          label_candidates_list.append(padded_label_candidates)
          label_candidates_mask_list.append(padded_label_candidates_mask)
        while len(label_candidates_list) < max_predicate_candidate_length:
          # We assue the predicate length is same as the original sentence length without BPE tokenization
          label_candidates_list.append([0] * max_label_candidate_length)
          label_candidates_mask_list.append([0] * max_label_candidate_length)
      else:
        label_candidates_list = None
        label_candidates_mask_list = None


      # Classical based feature process
      # _input_token_ids = example._input_token_ids + [0] * (_max_token_length - example._len)
      # _arg_candidates = example._enumerated_span_candidates + [
      #   (1, 0)
      # ] * (_max_arg_candidate_length - len(example._enumerated_span_candidates))
      # _predicate_candidates = example._predicate_candidates + [
      #   (1, 0)
      # ]* (_max_predicate_candidate_length - len(example._predicate_candidates))
      #
      # if label_format is None:
      #   _label_ids = None
      #
      # elif label_format == SRL_LABEL_SPAN_BASED:
      #   _label_ids = example._label_ids + [
      #     (1, 0, 1, 0, example.label_padding_id)
      #   ] * (_max_label_length - example._label_len)
      #
      # elif label_format == SRL_LABEL_SEQ_BASED:
      #   _label_ids = example._label_ids + [example.label_padding_id] * (
      #     _max_label_length - example._label_len)
      # else:
      #   raise ValueError(
      #     "The label format {} is not supported. Choice: {} | {}".format(
      #       label_format, SRL_LABEL_SEQ_BASED, SRL_LABEL_SPAN_BASED))

      features.append(
          SRLFeature(input_ids=input_ids,
                     input_mask=input_mask,
                     segment_ids=segment_ids,
                     is_head=is_head,
                     label_ids=label_ids,
                     arg_candidates=arg_candidates,
                     predicate_candidates=predicate_candidates,
                     arg_candidate_label_ids = arg_candidate_label_ids,
                     predicate_candidate_label_ids = predicate_candidate_label_ids,
                     label_candidates = label_candidates_list,
                     label_candidates_mask = label_candidates_mask_list,
                     pos_tag_label_ids=pos_tag_label_ids,
                     predicate_idx=predicate_idx,
                     _input_token_ids=None, # _input_token_ids,
                     _label_ids=None, #_label_ids,
                     _arg_candidates=None, # _arg_candidates,
                     _predicate_candidates=None, #_predicate_candidates,
                     _token_length=0# example._len
                     ))
    return features

  def decode_to_labels(self, preds, mb: MiniBatch):
    labels = []
    for example, pred_logits in zip(mb.examples, preds[mb.task_name]["logits"]):
      preds_tags = pred_logits.argmax(-1).data.cpu().numpy()
      span_preds, pred_labels = get_span_labels(
        sentence_tags=preds_tags,
        is_head=example.is_head,
        segment_id=example.segment_ids,
        inv_label_mapping=self._inv_label_mapping)
      span_preds = set(filter(lambda item: item[0] != example.predicate_index, span_preds))
      # Because the get_span_labels is inclusive
      span_preds = [(x[0], x[1]+1, x[2]) for x in span_preds]
      labels.append(span_preds)
    return labels