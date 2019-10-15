from typing import List

import torch

from relogic.logickit.dataflow import DataFlow, Example, Feature, MiniBatch
from relogic.logickit.tokenizer.tokenization import BertTokenizer
from relogic.logickit.utils import create_tensor, truncate_seq_pair


class PairwiseExample(Example):
  """
  text (str): query text
  text_p (str): positive example
  text_n (str): negative example
  """
  def __init__(self, guid: str, text: str, p_guid: str=None, text_p:str=None, n_guid: str=None, text_n:str=None):
    super(PairwiseExample, self).__init__()
    self.guid = guid
    self.text = text
    self.p_guid = p_guid
    self.text_p = text_p
    self.n_guid = n_guid
    self.text_n = text_n

  def process(self, tokenizers, *inputs, **kwargs):
    for tokenizer in tokenizers.values():
      if isinstance(tokenizer, BertTokenizer):
        max_seq_length = kwargs.pop("max_seq_length", 512)

        self.text_tokens, self.text_is_head = tokenizer.tokenize(self.text)
        self.text_tokens = self.text_tokens[:max_seq_length - 2]
        self.text_is_head = self.text_is_head[:max_seq_length - 2]
        self.tokens = ["[CLS]"] + self.text_tokens + ["[SEP]"]
        self.input_ids = tokenizer.convert_tokens_to_ids(self.tokens)
        self.input_mask = [1] * len(self.input_ids)
        self.segment_ids = [0] * len(self.input_ids)

        if self.text_p:
          self.text_p_tokens, self.text_p_is_head = tokenizer.tokenize(self.text_p)
          self.text_p_tokens = self.text_p_tokens[:max_seq_length - 2]
          self.text_p_is_head = self.text_p_is_head[:max_seq_length - 2]
          self.p_tokens = ["[CLS]"] + self.text_p_tokens + ["[SEP]"]
          self.p_input_ids = tokenizer.convert_tokens_to_ids(self.p_tokens)
          self.p_input_mask = [1] * len(self.p_input_ids)
          self.p_segment_ids = [0] * len(self.p_input_ids)

        if self.text_n:
          self.text_n_tokens, self.text_n_is_head = tokenizer.tokenize(self.text_n)
          self.text_n_tokens = self.text_n_tokens[:max_seq_length - 2]
          self.text_n_is_head = self.text_n_is_head[:max_seq_length - 2]
          self.n_tokens = ["[CLS]"] + self.text_n_tokens + ["[SEP]"]
          self.n_input_ids = tokenizer.convert_tokens_to_ids(self.n_tokens)
          self.n_input_mask = [1] * len(self.n_input_ids)
          self.n_segment_ids = [0] * len(self.n_input_ids)

  @classmethod
  def from_structure(cls, structure):
    return cls(guid="", text=structure.text)

  @classmethod
  def from_json(cls, example):
    return cls(guid=example.get("guid", None),
               text=example["text"],
               text_p=example.get("text_p", None),
               text_n=example.get("text_n", None),
               p_guid=example.get("p_guid", None),
               n_guid=example.get("n_guid", None))

  @property
  def len(self):
    return len(self.tokens)

  @property
  def len_p(self):
    return len(self.p_tokens)

  @property
  def len_n(self):
    return len(self.n_tokens)

class PairwiseFeature(Feature):
  """

  """
  def __init__(self, *inputs, **kwargs):
    super(PairwiseFeature, self).__init__()
    # BERT based feature
    self.input_ids = kwargs.pop("input_ids")
    self.input_mask = kwargs.pop("input_mask")
    self.segment_ids = kwargs.pop("segment_ids")
    self.p_input_ids = kwargs.pop("p_input_ids", None)
    self.p_input_mask = kwargs.pop("p_input_mask", None)
    self.p_segment_ids = kwargs.pop("p_segment_ids", None)
    self.n_input_ids = kwargs.pop("n_input_ids", None)
    self.n_input_mask = kwargs.pop("n_input_mask", None)
    self.n_segment_ids = kwargs.pop("n_segment_ids", None)

class PairwiseMiniBatch(MiniBatch):
  def __init__(self, *inputs, **kwargs):
    super(PairwiseMiniBatch, self).__init__(*inputs, **kwargs)

  def generate_input(self, device, use_label):
    inputs = {}
    inputs["task_name"] = self.task_name
    inputs["input_ids"] = create_tensor(self.input_features, "input_ids",
                                        torch.long, device)
    inputs["input_mask"] = create_tensor(self.input_features, "input_mask",
                                        torch.long, device)
    inputs["segment_ids"] = create_tensor(self.input_features, "segment_ids",
                                          torch.long, device)
    inputs["p_input_ids"] = create_tensor(self.input_features, "p_input_ids",
                                          torch.long, device)
    inputs["p_input_mask"] = create_tensor(self.input_features, "p_input_mask",
                                           torch.long, device)
    inputs["p_segment_ids"] = create_tensor(self.input_features, "p_segment_ids",
                                            torch.long, device)
    inputs["n_input_ids"] = create_tensor(self.input_features, "n_input_ids",
                                          torch.long, device)
    inputs["n_input_mask"] = create_tensor(self.input_features, "n_input_mask",
                                           torch.long, device)
    inputs["n_segment_ids"] = create_tensor(self.input_features, "n_segment_ids",
                                            torch.long, device)
    inputs["is_inference"] = not use_label
    inputs["extra_args"] = {"target": -torch.ones(inputs["input_ids"].size(0)).to(device)}

    return inputs

class PairwiseDataFlow(DataFlow):
  def __init__(self, config, task_name, tokenizers, label_mapping=None):
    super(PairwiseDataFlow, self).__init__(config, task_name, tokenizers, label_mapping)

  @property
  def example_class(self):
    return PairwiseExample

  @property
  def minibatch_class(self):
    return PairwiseMiniBatch

  def process_example(self, example: PairwiseExample):
    example.process(tokenizers=self.tokenizers, max_seq_length=self.config.max_seq_length)

  def convert_examples_to_features(self, examples: List[PairwiseExample]):
    examples: List[PairwiseExample]
    features = []

    max_token_length = max([example.len for example in examples])
    try:
      max_token_length_p = max([example.len_p for example in examples])
      max_token_length_n = max([example.len_n for example in examples])
    except:
      max_token_length_p = 0
      max_token_length_n = 0

    for idx, example in enumerate(examples):
      padding = [0] * (max_token_length - example.len)
      input_ids = example.input_ids + padding
      input_mask = example.input_mask + padding
      segment_ids = example.segment_ids + padding

      if example.p_input_ids and example.n_input_ids:
        p_padding = [0] * (max_token_length_p - example.len_p)
        p_input_ids = example.p_input_ids + p_padding
        p_input_mask = example.p_input_mask + p_padding
        p_segment_ids = example.p_segment_ids + p_padding

        n_padding = [0] * (max_token_length_n - example.len_n)
        n_input_ids = example.n_input_ids + n_padding
        n_input_mask = example.n_input_mask + n_padding
        n_segment_ids = example.n_segment_ids + n_padding

        features.append(
          PairwiseFeature(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          p_input_ids=p_input_ids,
                          p_input_mask=p_input_mask,
                          p_segment_ids=p_segment_ids,
                          n_input_ids=n_input_ids,
                          n_input_mask=n_input_mask,
                          n_segment_ids=n_segment_ids))
      else:
        features.append(
          PairwiseFeature(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids))
    return features





