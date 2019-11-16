from typing import List, Tuple

import torch

from relogic.logickit.dataflow import DataFlow, Example, Feature, MiniBatch
from relogic.logickit.utils import create_tensor
from relogic.logickit.tokenizer import CustomizedBertTokenizer, RobertaXLMTokenizer


class MixSentExample(Example):
  """MixSent Example contains the attributes and functionality for
  MixSent Semi-supervised Learning.

  """
  def __init__(
        self, text_a: str, text_b: str, text_c: str,
        span_a, span_b, span_c_a, span_c_b):
    super().__init__()
    self.text_a = text_a
    self.a_raw_tokens = text_a.split()
    self.text_b = text_b
    self.b_raw_tokens = text_b.split()
    self.text_c = text_c
    self.c_raw_tokens = text_c.split()
    self.span_a = span_a
    self.span_b = span_b
    self.span_c_a = span_c_a
    self.span_c_b = span_c_b

    self.padding_id = 0

  def process(self, tokenizers, *inputs, **kwargs):
    for tokenizer in tokenizers.values():
      if isinstance(tokenizer, RobertaXLMTokenizer):
        (self.a_tokens, self.a_is_head,
         self.a_input_ids) = tokenizer.tokenize_and_add_placeholder_and_convert_to_ids(self.text_a, self.a_raw_tokens)
        (self.b_tokens, self.b_is_head,
         self.b_input_ids) = tokenizer.tokenize_and_add_placeholder_and_convert_to_ids(self.text_b, self.b_raw_tokens)
        (self.c_tokens, self.c_is_head,
         self.c_input_ids) = tokenizer.tokenize_and_add_placeholder_and_convert_to_ids(self.text_c, self.c_raw_tokens)

        self.a_segment_ids = [0] * len(self.a_tokens)
        self.b_segment_ids = [0] * len(self.b_tokens)
        self.c_segment_ids = [0] * len(self.c_tokens)

        self.a_head_index = [idx for idx, value in enumerate(self.a_is_head) if value == 1]
        self.b_head_index = [idx for idx, value in enumerate(self.b_is_head) if value == 1]
        self.c_head_index = [idx for idx, value in enumerate(self.c_is_head) if value == 1]

        self.a_input_mask = [1] * len(self.a_input_ids)
        self.b_input_mask = [1] * len(self.b_input_ids)
        self.c_input_mask = [1] * len(self.c_input_ids)

        self.span_a_selected_index = [self.a_head_index[index] for index in range(self.span_a[0], self.span_a[1])]
        self.span_b_selected_index = [self.b_head_index[index] for index in range(self.span_b[0], self.span_b[1])]
        self.span_c_a_selected_index = [self.c_head_index[index] for index in range(self.span_c_a[0], self.span_c_a[1])]
        self.span_c_b_selected_index = [self.c_head_index[index] for index in range(self.span_c_b[0], self.span_c_b[1])]

        self.padding_id = 1

  @classmethod
  def from_structure(cls, structure):
    pass

  @classmethod
  def from_json(cls, example):
    if isinstance(example["text_a"], list):
      # TODO depends on the language!
      text_a = " ".join(example["text_a"])
      text_b = " ".join(example["text_b"])
      text_c = " ".join(example["text_c"])
    elif isinstance(example["text_a"], str):
      text_a = example["text_a"]
      text_b = example["text_b"]
      text_c = example["text_c"]
    else:
      raise NotImplementedError("Unsupported data type {}".format(type(example["text_a"])))

    return cls(text_a=text_a,
               text_b=text_b,
               text_c=text_c,
               span_a=example["span_a"],
               span_b=example["span_b"],
               span_c_a=example["span_c_a"],
               span_c_b=example["span_c_b"])

  @property
  def len(self):
    return len(self.a_input_ids) + len(self.b_input_ids) + len(self.c_input_ids)

  @property
  def len_a(self):
    return len(self.a_input_ids)

  @property
  def len_b(self):
    return len(self.b_input_ids)

  @property
  def len_c(self):
    return len(self.c_input_ids)

class MixSentFeature(Feature):
  """

  """
  def __init__(self, *inputs, **kwargs):
    super().__init__()
    self.a_input_ids = kwargs.pop("a_input_ids")
    self.b_input_ids = kwargs.pop("b_input_ids")
    self.c_input_ids = kwargs.pop("c_input_ids")
    self.a_segment_ids = kwargs.pop("a_segment_ids")
    self.b_segment_ids = kwargs.pop("b_segment_ids")
    self.c_segment_ids = kwargs.pop("c_segment_ids")
    self.a_is_head = kwargs.pop("a_is_head")
    self.b_is_head = kwargs.pop("b_is_head")
    self.c_is_head = kwargs.pop("c_is_head")
    self.a_input_mask = kwargs.pop("a_input_mask")
    self.b_input_mask = kwargs.pop("b_input_mask")
    self.c_input_mask = kwargs.pop("c_input_mask")
    self.span_a_selected_index = kwargs.pop("span_a_selected_index")
    self.span_b_selected_index = kwargs.pop("span_b_selected_index")
    self.span_c_a_selected_index = kwargs.pop("span_c_a_selected_index")
    self.span_c_b_selected_index = kwargs.pop("span_c_b_selected_index")

class MixSentMiniBatch(MiniBatch):

  def __init__(self, *inputs, **kwargs):
    super().__init__(*inputs, **kwargs)

  def generate_input(self, device, use_label):
    inputs = {}
    inputs["task_name"] = self.task_name
    inputs["a_input_ids"] = create_tensor(self.input_features, "a_input_ids",
                                        torch.long, device)
    inputs["a_input_mask"] = create_tensor(self.input_features, "a_input_mask",
                                         torch.long, device)
    inputs["a_segment_ids"] = create_tensor(self.input_features, "a_segment_ids",
                                         torch.long, device)
    inputs["a_input_head"] = create_tensor(self.input_features, "a_is_head",
                                           torch.long, device)
    inputs["b_input_ids"] = create_tensor(self.input_features, "b_input_ids",
                                          torch.long, device)
    inputs["b_input_mask"] = create_tensor(self.input_features, "b_input_mask",
                                           torch.long, device)
    inputs["b_segment_ids"] = create_tensor(self.input_features, "b_segment_ids",
                                            torch.long, device)
    inputs["b_input_head"] = create_tensor(self.input_features, "b_is_head",
                                           torch.long, device)
    inputs["c_input_ids"] = create_tensor(self.input_features, "c_input_ids",
                                          torch.long, device)
    inputs["c_input_mask"] = create_tensor(self.input_features, "c_input_mask",
                                           torch.long, device)
    inputs["c_segment_ids"] = create_tensor(self.input_features, "c_segment_ids",
                                            torch.long, device)
    inputs["c_input_head"] = create_tensor(self.input_features, "c_is_head",
                                           torch.long, device)
    inputs["span_a_selected_index"] = create_tensor(self.input_features, "span_a_selected_index",
                                                    torch.long, device)
    inputs["span_b_selected_index"] = create_tensor(self.input_features, "span_b_selected_index",
                                                    torch.long, device)
    inputs["span_c_a_selected_index"] = create_tensor(self.input_features, "span_c_a_selected_index",
                                                    torch.long, device)
    inputs["span_c_b_selected_index"] = create_tensor(self.input_features, "span_c_b_selected_index",
                                                      torch.long, device)
    inputs["extra_args"] = {}

    return inputs

class MixSentDataFlow(DataFlow):
  def __init__(self, config, task_name, tokenizers, label_mapping):
    super().__init__(config, task_name, tokenizers, label_mapping)

  @property
  def example_class(self):
    return MixSentExample

  @property
  def minibatch_class(self):
    return MixSentMiniBatch

  def process_example(self, example: MixSentExample):
    example.process(tokenizers=self.tokenizers)

  def convert_examples_to_features(self, examples: List[MixSentExample]):
    examples: List[MixSentExample]
    features = []

    a_max_token_length = max([example.len_a for example in examples])
    b_max_token_length = max([example.len_b for example in examples])
    c_max_token_length = max([example.len_c for example in examples])

    span_a_selected_index_max_length = max([len(example.span_a_selected_index) for example in examples])
    span_b_selected_index_max_length = max([len(example.span_b_selected_index) for example in examples])
    span_c_a_selected_index_max_length = max([len(example.span_c_a_selected_index) for example in examples])
    span_c_b_selected_index_max_length = max([len(example.span_c_b_selected_index) for example in examples])

    for idx, example in enumerate(examples):
      a_padding = [0] * (a_max_token_length - example.len_a)
      b_padding = [0] * (b_max_token_length - example.len_b)
      c_padding = [0] * (c_max_token_length - example.len_c)
      a_input_ids = example.a_input_ids + [example.padding_id] * (a_max_token_length - example.len_a)
      b_input_ids = example.b_input_ids + [example.padding_id] * (b_max_token_length - example.len_b)
      c_input_ids = example.c_input_ids + [example.padding_id] * (c_max_token_length - example.len_c)

      a_segment_ids = example.a_segment_ids + a_padding
      b_segment_ids = example.b_segment_ids + b_padding
      c_segment_ids = example.c_segment_ids + c_padding

      a_input_mask = example.a_input_mask + a_padding
      b_input_mask = example.b_input_mask + b_padding
      c_input_mask = example.c_input_mask + c_padding

      a_is_head = example.a_is_head + a_padding
      b_is_head = example.b_is_head + b_padding
      c_is_head = example.c_is_head + c_padding

      span_a_selected_index = example.span_a_selected_index + [0] * (
        span_a_selected_index_max_length - len(example.span_a_selected_index))
      span_b_selected_index = example.span_b_selected_index + [0] * (
        span_b_selected_index_max_length - len(example.span_b_selected_index))
      span_c_a_selected_index = example.span_c_a_selected_index + [0] * (
        span_c_a_selected_index_max_length - len(example.span_c_a_selected_index))
      span_c_b_selected_index = example.span_c_b_selected_index + [0] * (
        span_c_b_selected_index_max_length - len(example.span_c_b_selected_index))

      features.append(
        MixSentFeature(
          a_input_ids=a_input_ids,
          b_input_ids=b_input_ids,
          c_input_ids=c_input_ids,
          a_input_mask=a_input_mask,
          b_input_mask=b_input_mask,
          c_input_mask=c_input_mask,
          a_segment_ids=a_segment_ids,
          b_segment_ids=b_segment_ids,
          c_segment_ids=c_segment_ids,
          a_is_head=a_is_head,
          b_is_head=b_is_head,
          c_is_head=c_is_head,
          span_a_selected_index=span_a_selected_index,
          span_b_selected_index=span_b_selected_index,
          span_c_a_selected_index=span_c_a_selected_index,
          span_c_b_selected_index=span_c_b_selected_index))

    return features
