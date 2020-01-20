from typing import List, Tuple

import torch

from relogic.logickit.dataflow import DataFlow, Example, Feature, MiniBatch
from relogic.logickit.utils import create_tensor
from transformers.tokenization_utils import PreTrainedTokenizer



class TwinsExample(Example):
  """
  Args:
    text_a (str): sentence in source language
    text_b (str): sentence in target language
    label (str)
  """
  def __init__(self, guid: str, text_a: str, text_b: str, labels: str = None):
    super().__init__()
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.labels = labels
    self.padding_id = 0

  def process(self, tokenizers, *inputs, **kwargs):
    """Process the sentence pair."""

    for tokenizer in tokenizers.values():
      if isinstance(tokenizer, PreTrainedTokenizer):
        self.text_a_tokens = tokenizer.tokenize(self.text_a)
        self.text_b_tokens = tokenizer.tokenize(self.text_b)

        self.a_tokens = ["[CLS]"] + self.text_a_tokens + ["[SEP]"]
        self.a_segment_ids = [0] * len(self.a_tokens)

        self.b_tokens = ["[CLS]"] + self.text_b_tokens + ["[SEP]"]
        self.b_segment_ids = [0] * len(self.b_tokens)

        self.a_input_ids = tokenizer.convert_tokens_to_ids(self.a_tokens)
        self.b_input_ids = tokenizer.convert_tokens_to_ids(self.b_tokens)
        self.a_input_mask = [1] * len(self.a_input_ids)
        self.b_input_mask = [1] * len(self.b_input_ids)

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
    return cls(guid="",
               text_a=structure.text_a,
               text_b=structure.text_b)

  @classmethod
  def from_json(cls, example):
    return cls(guid="{}|{}".format(example.get("text_a_id", 0), example.get("text_b_id", 0)),
               text_a=example["text_a"],
               text_b=example["text_b"],
               labels=example.get("label", None))

  @property
  def len(self):
    return len(self.a_tokens) + len(self.b_tokens)

  @property
  def len_a(self):
    return len(self.a_tokens)

  @property
  def len_b(self):
    return len(self.b_tokens)

class TwinsFeature(Feature):
  """Parallel Feature

  """
  def __init__(self, *input, **kwargs):
    super().__init__()
    self.a_input_ids = kwargs.pop("a_input_ids")
    self.b_input_ids = kwargs.pop("b_input_ids")
    self.a_segment_ids = kwargs.pop("a_segment_ids")
    self.b_segment_ids = kwargs.pop("b_segment_ids")
    self.a_input_mask = kwargs.pop("a_input_mask")
    self.b_input_mask = kwargs.pop("b_input_mask")
    self.label_ids = kwargs.pop("label_ids")

class TwinsMiniBatch(MiniBatch):
  def __init__(self, *inputs, **kwargs):
    super().__init__(*inputs, **kwargs)

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

    if use_label:
      label_ids = create_tensor(self.input_features, "label_ids",
                                torch.long, device)
      inputs["label_ids"] = label_ids
    else:
      inputs["label_ids"] = None


    inputs["extra_args"] = {}
    if self.config.tasks[self.task_name]["selected_non_final_layers"] is not None:
      inputs["extra_args"]["selected_non_final_layers"] = self.config.tasks[self.task_name]["selected_non_final_layers"]


    return inputs

class TwinsDataFlow(DataFlow):
  """DataFlow implementation based on Parallel Task"""
  def __init__(self, config, task_name, tokenizers, label_mapping):
    super().__init__(config, task_name, tokenizers, label_mapping)

  @property
  def example_class(self):
    return TwinsExample

  @property
  def minibatch_class(self):
    return TwinsMiniBatch

  def process_example(self, example: TwinsExample):
    example.process(tokenizers=self.tokenizers, label_mapping=self.label_mapping)

  def convert_examples_to_features(self, examples: List[TwinsExample]):
    examples: List[TwinsExample]
    features = []
    a_max_token_length = max([example.len_a for example in examples])
    b_max_token_length = max([example.len_b for example in examples])

    for idx, example in enumerate(examples):
      a_padding = [0] * (a_max_token_length - example.len_a)
      b_padding = [0] * (b_max_token_length - example.len_b)
      a_input_ids = example.a_input_ids + [example.padding_id] * (a_max_token_length - example.len_a)
      b_input_ids = example.b_input_ids + [example.padding_id] * (b_max_token_length - example.len_b)
      a_segment_ids = example.a_segment_ids + a_padding
      b_segment_ids = example.b_segment_ids + b_padding
      a_input_mask = example.a_input_mask + a_padding
      b_input_mask = example.b_input_mask + b_padding

      if example.label_ids is not None:
        # We assume the label length is same as sequence length
        label_ids = example.label_ids
      else:
        label_ids = None

      features.append(
        TwinsFeature(
          a_input_ids=a_input_ids,
          b_input_ids=b_input_ids,
          a_input_mask=a_input_mask,
          b_input_mask=b_input_mask,
          a_segment_ids=a_segment_ids,
          b_segment_ids=b_segment_ids,
          label_ids=label_ids))
    return features