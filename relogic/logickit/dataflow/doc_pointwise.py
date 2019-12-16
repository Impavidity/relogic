from typing import List

import torch

from relogic.logickit.dataflow import DataFlow, Example, Feature, MiniBatch
from relogic.logickit.tokenizer.tokenization import BertTokenizer
from relogic.logickit.utils import create_tensor, truncate_seq_pair


class DocPointwiseExample(Example):
  """

  """
  def __init__(self, guid: str, text_a: str, text_bs: List[str], label=None, sent_label=None):
    super().__init__()
    self.guid = guid
    self.text_a = text_a
    self.text_bs = text_bs
    self.label = label
    self.sent_label = sent_label
    self.sent_num = len(self.text_bs)

  def process(self, tokenizers, *inputs, regression=False, **kwargs):
    for tokenizer in tokenizers.values():
      if isinstance(tokenizer, BertTokenizer):
        self.text_a_tokens, self.text_a_is_head = tokenizer.tokenize(self.text_a)
        self.text_bs_tokens, self.text_bs_is_head = zip(*[tokenizer.tokenize(text_b) for text_b in self.text_bs])

        max_seq_length = kwargs.pop("max_seq_length", 512)
        for i in range(self.sent_num):
          truncate_seq_pair(self.text_a_tokens, self.text_bs_tokens[i], max_seq_length - 3)
          truncate_seq_pair(self.text_a_is_head, self.text_bs_is_head[i], max_seq_length - 3)
        self.tokens_list = []
        self.segment_ids_list = []
        self.is_head_list = []
        self.input_ids_list = []
        self.input_mask_list = []
        for i in range(self.sent_num):
          self.tokens_list.append(["[CLS]"] + self.text_a_tokens + ["[SEP]"] + self.text_bs_tokens[i] + ["[SEP]"])
          self.segment_ids_list.append([0] * (len(self.text_a_tokens) + 2) + [1] * (len(self.text_bs_tokens[i]) + 1))
          self.is_head_list.append([2] + self.text_a_is_head + [2] + self.text_bs_is_head[i] + [2])

          self.input_ids_list.append(tokenizer.convert_tokens_to_ids(self.tokens_list[i]))
          self.input_mask_list.append([1] * len(self.input_ids_list[i]))

        if self.label is not None:
          if regression:
            # Regression Problem with sigmoid loss function
            self.label_ids = float(self.label)
          else:
            label_mapping = kwargs.get("label_mapping")
            self.label_ids = label_mapping[self.label]

  @classmethod
  def from_structure(cls, structure):
    return cls(guid="", text_a=structure.text_a, text_bs=structure.text_b)

  @classmethod
  def from_json(cls, example):
    return cls(guid="{}|{}".format(example.get("text_a_id", 0), example.get("text_b_id", 0)),
               text_a=example["text_a"],
               text_bs=example["text_b"],
               label=example.get("label", None))

  @property
  def len(self):
    return 1

  def _len(self, i):
    return len(self.input_ids_list[i])

class DocPointwiseFeature(Feature):
  def __init__(self, *inputs, **kwargs):
    super().__init__()
    self.input_ids = kwargs.pop("input_ids")
    self.input_mask = kwargs.pop("input_mask")
    self.segment_ids = kwargs.pop("segment_ids")
    self.label_ids = kwargs.pop("label_ids")

class DocPointwiseMiniBatch(MiniBatch):
  def __init__(self, *inputs, **kwargs):
    super().__init__(*inputs, **kwargs)

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
      if self.config.regression:
        inputs["label_ids"] = create_tensor(self.input_features, "label_ids",
                                            torch.float, device)
      else:
        inputs["label_ids"] = create_tensor(self.input_features, "label_ids",
                                            torch.long, device)
    else:
      inputs["label_ids"] = None
    inputs["extra_args"] = {}
    return inputs

class DocPointwiseDataFlow(DataFlow):
  def __init__(self, config, task_name, tokenizers, label_mapping):
    super().__init__(config, task_name, tokenizers, label_mapping)

  @property
  def example_class(self):
    return DocPointwiseExample

  @property
  def minibatch_class(self):
    return DocPointwiseMiniBatch

  def process_example(self, example: DocPointwiseExample):
    example.process(tokenizers=self.tokenizers,
                    label_mapping=self.label_mapping,
                    max_seq_length=self.config.max_seq_length,
                    regression=self.config.regression)

  def convert_examples_to_features(self, examples: List[DocPointwiseExample]):
    examples: List[DocPointwiseExample]
    # Because we want to have batch operation on one example
    # So len(examples) == 1
    features = []
    assert len(examples) == 1, "We can only handle one example at a time for batching"
    example = examples[0]
    max_token_length = max([example._len(i) for i in range(example.sent_num)])

    for idx in range(example.sent_num):
      padding = [0] * (max_token_length - example._len(idx))
      input_ids = example.input_ids_list[idx] + padding
      input_mask = example.input_mask_list[idx] + padding
      segment_ids = example.segment_ids_list[idx] + padding
      if hasattr(example, "label_ids"):
        label_ids = example.label_ids
      else:
        label_ids = None

      features.append(
        DocPointwiseFeature(input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         label_ids=label_ids))
    return features

  def decode_to_labels(self, preds, mbs: DocPointwiseMiniBatch):
    return None