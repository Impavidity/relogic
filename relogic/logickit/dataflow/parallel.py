from typing import List, Tuple

import torch

from relogic.logickit.dataflow import DataFlow, Example, Feature, MiniBatch
from relogic.logickit.utils import create_tensor
from relogic.logickit.tokenizer import CustomizedBertTokenizer, RobertaXLMTokenizer



class ParallelExample(Example):
  """Parallel Example contains the attributes and functionality of
  a parallel sentence pair. This class is mainly used for auxiliary
  task of multi-lingual model training. Parallel sentence (in different
  language) should be provided and token level alignment is needed.

  Args:
    text_a (str): sentence in source language
    text_b (str): sentence in target language
    alignment (Tuple[List]): token level alignment. We plan to take the format
      that tuple of list. There are two list in the tuple, first one is
      the index for source sentence, second one is the index for target
      sentence, which are parallel in the corresponding position.
      If you have the data format same as fast_align, you can use the scripts
      to convert into this format. The conversion rules are described in the script.
  """
  def __init__(self, text_a: str, text_b: str, alignment: Tuple[List]):
    super(ParallelExample, self).__init__()
    self.text_a = text_a
    self.text_b = text_b
    self.alignment = alignment
    self.padding_id = 0

  def process(self, tokenizers, *inputs, **kwargs):
    """Process the sentence pair."""

    for tokenizer in tokenizers.values():
      if isinstance(tokenizer, CustomizedBertTokenizer):
        self.text_a_tokens, self.text_a_is_head = tokenizer.tokenize(self.text_a)
        self.text_b_tokens, self.text_b_is_head = tokenizer.tokenize(self.text_b)

        max_seq_length = kwargs.pop("max_seq_length", 512)
        self.text_a_tokens = self.text_a_tokens[:max_seq_length-2]
        self.text_b_tokens = self.text_b_tokens[:max_seq_length-2]
        # We need to keep position for [CLS] and [SEP]

        self.a_tokens = ["[CLS]"] + self.text_a_tokens + ["[SEP]"]
        self.a_segment_ids = [0] * len(self.a_tokens)
        self.a_is_head = [2] + self.text_a_is_head + [2]

        self.b_tokens = ["[CLS]"] + self.text_b_tokens + ["[SEP]"]
        self.b_segment_ids = [0] * len(self.b_tokens)
        self.b_is_head = [2] + self.text_b_is_head + [2]

        self.a_input_ids = tokenizer.convert_tokens_to_ids(self.a_tokens)
        self.b_input_ids = tokenizer.convert_tokens_to_ids(self.b_tokens)
        self.a_input_mask = [1] * len(self.a_input_ids)
        self.b_input_mask = [1] * len(self.b_input_ids)

        self.a_head_index = [
          idx for idx, value in enumerate(self.a_is_head) if value == 1
        ] + [len(self.a_is_head) - 1]
        self.b_head_index = [
          idx for idx, value in enumerate(self.b_is_head) if value == 1
        ] + [len(self.b_is_head) - 1]

        if self.alignment is not None:
          self.a_selected_indices = [self.a_head_index[index] for index in self.alignment[0]]
          self.b_selected_indices = [self.b_head_index[index] for index in self.alignment[1]]
        else:
          self.a_selected_indices = None
          self.b_selected_indices = None
      elif isinstance(tokenizer, RobertaXLMTokenizer):
        self.padding_id = 1
        (self.a_tokens, self.a_is_head,
         self.a_input_ids) = tokenizer.tokenize_and_add_placeholder_and_convert_to_ids(
          self.text_a, None)
        (self.b_tokens, self.b_is_head,
         self.b_input_ids) = tokenizer.tokenize_and_add_placeholder_and_convert_to_ids(
          self.text_b, None)
        self.a_segment_ids = [0] * len(self.a_tokens)
        self.b_segment_ids = [0] * len(self.b_tokens)
        self.a_input_mask = [1] * len(self.a_input_ids)
        self.b_input_mask = [1] * len(self.b_input_ids)

        # self.a_head_index = [
        #                       idx for idx, value in enumerate(self.a_is_head) if value == 1
        #                     ] + [len(self.a_is_head) - 1]
        # self.b_head_index = [
        #                       idx for idx, value in enumerate(self.b_is_head) if value == 1
        #                     ] + [len(self.b_is_head) - 1]
        #
        # if self.alignment is not None:
        #   self.a_selected_indices = [self.a_head_index[index] for index in self.alignment[0]]
        #   self.b_selected_indices = [self.b_head_index[index] for index in self.alignment[1]]
        # else:
        self.a_selected_indices = None
        self.b_selected_indices = None




  @classmethod
  def from_structure(cls, structure):
    return cls(text_a=structure.text_a,
               text_b=structure.text_b,
               alignment=structure.alignment)

  @classmethod
  def from_json(cls, example):
    return cls(text_a=example["text_a"],
               text_b=example["text_b"],
               alignment=example.get("alignment", None))

  @property
  def len(self):
    return len(self.a_tokens) + len(self.b_tokens)

  @property
  def len_a(self):
    return len(self.a_tokens)

  @property
  def len_b(self):
    return len(self.b_tokens)

class ParallelFeature(Feature):
  """Parallel Feature

  """
  def __init__(self, *input, **kwargs):
    super(ParallelFeature, self).__init__()
    self.a_input_ids = kwargs.pop("a_input_ids")
    self.b_input_ids = kwargs.pop("b_input_ids")
    self.a_segment_ids = kwargs.pop("a_segment_ids")
    self.b_segment_ids = kwargs.pop("b_segment_ids")
    self.a_input_mask = kwargs.pop("a_input_mask")
    self.b_input_mask = kwargs.pop("b_input_mask")
    self.a_selected_indices = kwargs.pop("a_selected_indices")
    self.b_selected_indices = kwargs.pop("b_selected_indices")
    self.a_is_head = kwargs.pop("a_is_head")
    self.b_is_head = kwargs.pop("b_is_head")

class ParallelMiniBatch(MiniBatch):
  def __init__(self, *inputs, **kwargs):
    super(ParallelMiniBatch, self).__init__(*inputs, **kwargs)

  def generate_input(self, device, use_label):
    """Generate tensors based on Parallel Feature"""
    # BERT based features
    inputs = {}
    inputs["task_name"] = self.task_name
    inputs["a_input_ids"] = create_tensor(self.input_features, "a_input_ids",
                                          torch.long, device)
    inputs["b_input_ids"] = create_tensor(self.input_features, "b_input_ids",
                                          torch.long, device)
    inputs["a_input_mask"] = create_tensor(self.input_features, "a_input_mask",
                                           torch.long, device)
    inputs["b_input_mask"] = create_tensor(self.input_features, "b_input_mask",
                                           torch.long, device)
    inputs["a_segment_ids"] = create_tensor(self.input_features, "a_segment_ids",
                                            torch.long, device)
    inputs["b_segment_ids"] = create_tensor(self.input_features, "b_segment_ids",
                                            torch.long, device)
    inputs["a_selected_indices"] = create_tensor(self.input_features, "a_selected_indices",
                                            torch.long, device)
    inputs["b_selected_indices"] = create_tensor(self.input_features, "b_selected_indices",
                                            torch.long, device)
    inputs["a_is_head"] = create_tensor(self.input_features, "a_is_head",
                                            torch.long, device)
    inputs["b_is_head"] = create_tensor(self.input_features, "b_is_head",
                                            torch.long, device)

    inputs["extra_args"] = {}
    if self.config.tasks[self.task_name]["selected_non_final_layers"] is not None:
      inputs["extra_args"]["selected_non_final_layers"] = self.config.tasks[self.task_name]["selected_non_final_layers"]


    return inputs

class ParallelDataFlow(DataFlow):
  """DataFlow implementation based on Parallel Task"""
  def __init__(self, config, task_name, tokenizers, label_mapping):
    super(ParallelDataFlow, self).__init__(config, task_name, tokenizers, label_mapping)

  @property
  def example_class(self):
    return ParallelExample

  @property
  def minibatch_class(self):
    return ParallelMiniBatch

  def process_example(self, example: ParallelExample):
    example.process(tokenizers=self.tokenizers)

  def convert_examples_to_features(self, examples: List[ParallelExample]):
    examples: List[ParallelExample]
    features = []
    a_max_token_length = max([example.len_a for example in examples])
    b_max_token_length = max([example.len_b for example in examples])

    try:
      max_selected_indices_length = max([len(example.a_selected_indices) for example in examples])
    except:
      max_selected_indices_length = 0

    for idx, example in enumerate(examples):
      a_padding = [0] * (a_max_token_length - example.len_a)
      b_padding = [0] * (b_max_token_length - example.len_b)
      a_input_ids = example.a_input_ids + [example.padding_id] * (a_max_token_length - example.len_a)
      b_input_ids = example.b_input_ids + [example.padding_id] * (b_max_token_length - example.len_b)
      a_segment_ids = example.a_segment_ids + a_padding
      b_segment_ids = example.b_segment_ids + b_padding
      a_input_mask = example.a_input_mask + a_padding
      b_input_mask = example.b_input_mask + b_padding
      a_is_head = example.a_is_head + a_padding
      b_is_head = example.b_is_head + b_padding

      if max_selected_indices_length > 0:
        a_selected_indices = example.a_selected_indices + [0] * (
            max_selected_indices_length - len(example.a_selected_indices))
        b_selected_indices = example.b_selected_indices + [0] * (
            max_selected_indices_length - len(example.b_selected_indices))
      else:
        a_selected_indices = None
        b_selected_indices = None

      features.append(
        ParallelFeature(
          a_input_ids=a_input_ids,
          b_input_ids=b_input_ids,
          a_input_mask=a_input_mask,
          b_input_mask=b_input_mask,
          a_segment_ids=a_segment_ids,
          b_segment_ids=b_segment_ids,
          a_selected_indices=a_selected_indices,
          b_selected_indices=b_selected_indices,
          a_is_head=a_is_head,
          b_is_head=b_is_head))
    return features