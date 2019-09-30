from typing import List

import torch

from relogic.logickit.dataflow import DataFlow, Example, Feature, MiniBatch
from relogic.logickit.tokenizer.tokenization import BertTokenizer
from relogic.logickit.utils import create_tensor



class ECPExample(Example):
  """EXP Example contains the attributes and functionality of
  a Emotion-Cause Pair example.

  Currently the implementation is based on Chinese.

  Args:
    text (str): A document string.
    label (List[Tuple]): A list of labels. According to the statistics
      of the data, most(89%) of the examples only contains one emotion-cause
      pair.

  """
  def __init__(self, text, clause_spans=None, labels=None):
    super(ECPExample, self).__init__()
    self.text = text
    self.clause_spans = clause_spans
    self.raw_tokens = text.split()
    self.labels = labels
    self.label_padding = '0'

  def process(self, tokenizers, *inputs, **kwargs):
    """Process the ECP example.

    This process requires the tokenizers.

    :param tokenizers:
    :param inputs:
    :param kwargs:

    """
    for tokenizer in tokenizers.values():
      if isinstance(tokenizer, BertTokenizer):
        # BERT process part
        self.text_tokens, _ = tokenizer.tokenize(self.text)

        # Some problem with Chinese Indexing
        # for token list (Chinese characters), they will be combined into a string without space.
        # Until now the spans index is still correct.
        # But after tokenization, something goes wrong. Because some numbers will be
        # merged (several digits will be merged)
        # Then the index information is wrong here.
        self.tokens = ["[CLS]"] + self.text_tokens + ["[SEP]"]
        self.char_index_to_token_index = []

        for idx, token in enumerate(self.tokens[1:]):
          # Because we use exclusive representation, so we process extra [SEP] token for padding
          self.char_index_to_token_index.extend([idx+1] * len(token))

        self.segment_ids = [0] * (len(self.text_tokens) + 2)

        self.input_ids = tokenizer.convert_tokens_to_ids(self.tokens)
        self.input_mask = [1] * len(self.input_ids)

        if self.labels is not None:
          label_mapping = kwargs.get("label_mapping")
          self.label_padding_id = label_mapping[self.label_padding]
          self.label_ids = []
          for label in self.labels:
            # Because we add [SEP] in front of each sentence, so we need to shift
            # the index to right for one step
            index_shift_right_label = tuple([self.char_index_to_token_index[item] for item in label])
            label_tuple = index_shift_right_label + (label_mapping["1"],)
            self.label_ids.append(label_tuple)

  @classmethod
  def from_structure(cls, structure):
    """Implementation of converting structure into ECPExample.
    TODO: consider how to automatically generate the clause spans"""

    return cls(text=structure.text)

  @classmethod
  def from_json(cls, example):
    """Implementation of converting json object into ECPExample.
    Currently we assume the language is Chinese. So there is
    no space between tokens. TODO: Make it general
    """
    return cls(text="".join(example["tokens"]),
               clause_spans=example.get("clause_spans", None),
               labels=example.get("labels", None))

  @property
  def len(self):
    return len(self.tokens)

  @property
  def label_len(self):
    if hasattr(self, "label_ids"):
      return len(self.label_ids)
    return 0


class ECPFeature(Feature):
  """ECP features.

  """
  def __init__(self, *inputs, **kwargs):
    super(ECPFeature, self).__init__()
    # BERT based feature
    self.input_ids = kwargs.pop("input_ids")
    self.input_mask = kwargs.pop("input_mask")
    self.segment_ids = kwargs.pop("segment_ids")
    self.label_ids = kwargs.pop("label_ids")
    self.clause_candidates = kwargs.pop("clause_candidates")


class ECPMiniBatch(MiniBatch):
  """Minibatch Implementation based on ECP task.

  """
  def __init__(self, *inputs, **kwargs):
    super(ECPMiniBatch, self).__init__(*inputs, **kwargs)

  def generate_input(self, device, use_label):
    """Generate tensors based on ECP Features"""
    # BERT based features
    inputs = {}
    inputs["task_name"] = self.task_name
    inputs["input_ids"] = create_tensor(self.input_features, "input_ids",
                                        torch.long, device)
    inputs["input_mask"] = create_tensor(self.input_features, "input_mask",
                                         torch.long, device)
    inputs["segment_ids"] = create_tensor(self.input_features, "segment_ids",
                                          torch.long, device)
    inputs["label_ids"] = create_tensor(self.input_features, "label_ids", 
                                          torch.long, device)
    inputs["clause_candidates"] = create_tensor(self.input_features, "clause_candidates",
                                          torch.long, device)
    inputs["extra_args"] = {}
    return inputs

class ECPDataFlow(DataFlow):
  """DataFlow implementation based on ECP task."""
  def __init__(self, config, task_name, tokenizers, label_mapping):
    super(ECPDataFlow, self).__init__(config, task_name, tokenizers, label_mapping)

  @property
  def example_class(self):
    return ECPExample

  @property
  def minibatch_class(self):
    return ECPMiniBatch

  def process_example(self, example: ECPExample):
    """Process ECP example"""
    example.process(tokenizers=self.tokenizers,
                    label_mapping=self.label_mapping)

  def convert_examples_to_features(self, examples: List[ECPExample]):
    examples: List[ECPExample]
    features = []

    # BERT based variables
    max_token_length = max([example.len for example in examples])
    max_clause_candidate_length = max([len(example.clause_spans) for example in examples])
    try:
      max_label_length = max([example.label_len for example in examples])
    except:
      max_label_length = None

    for idx, example in enumerate(examples):
      # BERT based feature process
      padding = [0] * (max_token_length - example.len)
      input_ids = example.input_ids + padding
      input_mask = example.input_mask + padding
      segment_ids = example.segment_ids + padding

      index_shift_right_clause_candidates = [(
        example.char_index_to_token_index[span[0]], example.char_index_to_token_index[span[1]])
        for span in example.clause_spans]
      clause_candidates = index_shift_right_clause_candidates + [
        (1, 0)
      ] * (max_clause_candidate_length - len(example.clause_spans))

      if example.label_ids is not None:
        label_ids = example.label_ids + [
          (1, 0, 1, 0, example.label_padding_id)
        ] * (max_label_length - example.label_len)
      else:
        label_ids = None

      features.append(
        ECPFeature(input_ids=input_ids,
                   input_mask=input_mask,
                   segment_ids=segment_ids,
                   label_ids=label_ids,
                   clause_candidates=clause_candidates))
    return features

