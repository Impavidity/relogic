from typing import List, Dict
import random
import torch

from relogic.logickit.dataflow import DataFlow, Example, Feature, MiniBatch
from relogic.logickit.tokenizer.tokenization import BertTokenizer
from relogic.logickit.utils import create_tensor, truncate_seq_pair
from relogic.logickit.dataflow.pointwise import PointwiseExample, PointwiseFeature, PointwiseMiniBatch

class DocExample(Example):
  """DocPairwiseExample is for document representation learning.


  """
  def __init__(self, guid: str, edge_data=None, label=None):
    super().__init__()
    self.guid = guid
    self.edge_data = edge_data
    # List[(src, tgt)]
    self.label = label

  def process(self, tokenizers, *inputs, **kwargs):
    if self.label is not None:
      label_mapping = kwargs.get("label_mapping")
      self.label_ids = label_mapping[self.label]

  @classmethod
  def from_structure(cls, structure):
    return cls(guid="", edge_data=structure.edge_data)

  @classmethod
  def from_json(cls, example):
    return cls(guid="{}|{}".format(example["text_a_id"], example["text_b_id"]),
               edge_data=example["edge_data"],
               label=example.get("label", None))


class DocPairwiseFeature(Feature):
  def __init__(self, *inputs, **kwargs):
    super().__init__()
    self.label_ids = kwargs.pop("label_ids")
    self.edge_data = kwargs.pop("edge_data")
    self.doc_span = kwargs.pop("doc_span")

class DocPairwiseDataFlow(DataFlow):
  """
  This dataflow is for controlling the document examples.
  The input is in sentence format.
  So this dataflow should have new way to organize the examples by rewriting
    the `update_with_structures` and `update_with_jsons`
  We assume the guid is in the format of "query_id|sent_id"
  """
  def __init__(self, config, task_name, tokenizers, label_mapping):
    super().__init__(config, task_name, tokenizers, label_mapping)
    self.positive_candidates = {}
    self.negative_candidates = {}
    self._size = 0

  @property
  def example_class(self):
    return PointwiseExample

  @property
  def minibatch_class(self):
    return PointwiseMiniBatch

  def process_example(self, example):
    example.process(tokenizers=self.tokenizers,
                    label_mapping=self.label_mapping,
                    regression=self.config.regression)

  def update_with_structures(self, structures):
    self.examples = {}
    for structure in structures:
      topic_id = structure.text_a_id
      doc_id = structure.text_b_id.rsplit("-", 1)[0]
      if topic_id not in self.examples:
        self.examples[topic_id] = {}
      if doc_id not in self.examples[topic_id]:
        self.examples[topic_id][doc_id] = []
      self.examples[topic_id][doc_id].append(self.example_class.from_structure(structure))
    from tqdm import tqdm
    for topic_id in tqdm(self.examples):
      for doc_id in self.examples[topic_id]:
        self.process_example(self.examples[topic_id][doc_id])

  def update_with_jsons(self, examples):
    self.examples = {}
    self.doc_examples = {}
    for example in examples:
      if example["type"] == "sent":
        topic_id = example["text_a_id"]
        doc_id = example["text_b_id"].rsplit("-", 1)[0]
        if topic_id not in self.examples:
          self.examples[topic_id] = {}
        if doc_id not in self.examples[topic_id]:
          self.examples[topic_id][doc_id] = []
        self.examples[topic_id][doc_id].append(self.example_class.from_json(example))

      elif example["type"] == "doc":
        topic_id, doc_id = example["text_a_id"], example["text_b_id"]
        if topic_id not in self.doc_examples:
          self.doc_examples[topic_id] = {}
          self.positive_candidates[topic_id] = []
          self.negative_candidates[topic_id] = []
        if example["label"] == "1":
          self.positive_candidates[topic_id].append(doc_id)
          self._size += 1
        else:
          self.negative_candidates[topic_id].append(doc_id)
        self.doc_examples[topic_id][doc_id] = DocExample.from_json(example)
      else:
        raise ValueError("Unknown data type {}".format(example["type"]))

    from tqdm import tqdm
    for topic_id in tqdm(self.examples):
      for doc_id in self.examples[topic_id]:
        for example in self.examples[topic_id][doc_id]:
          self.process_example(example)
        self.process_example(self.doc_examples[topic_id][doc_id])

  def get_minibatches(self, minibatch_size, sequential=True, bucket=False):
    """"""

    if sequential:
      # If it is sequential, we just follow the example one by one.
      # We do not care about the positive sampling and negative sampling.
      # This is the mode for evaluation definitely.
      ids = []
      for topic_id in self.examples.keys():
        for doc_id in self.examples[topic_id].keys():
          ids.append((topic_id, doc_id))
      ids.sort()
      index = 0
      while index < len(ids):

        yield self._make_minibatch(ids[index: index+minibatch_size])
        index += minibatch_size
    else:
      # There is no bucketing setting.
      # In this setting, we only use positive sampling and negative sampling
      # 1. select topic based on batch size
      # 2. select positive example and negative example for each topic
      positive_candidates = []
      for topic_id in self.positive_candidates:
        for doc_id in self.positive_candidates[topic_id]:
          positive_candidates.append((topic_id, doc_id))
      # To make it deterministic
      positive_candidates.sort()
      random.shuffle(positive_candidates)
      if minibatch_size % 2 == 1:
        minibatch_size = minibatch_size + 1
      index = 0
      while index < self._size:
        pos_indices = positive_candidates[index: index+(minibatch_size//2)]
        neg_indices = []
        for pos_idx in pos_indices:
          neg_idx = random.sample(self.negative_candidates[pos_idx[0]], 1)[0]
          neg_indices.append(
            (pos_idx[0], neg_idx))

        yield self._make_minibatch(pos_indices + neg_indices)
        index += (minibatch_size // 2)

  def _make_minibatch(self, ids):
    examples = [self.examples[topic_id][doc_id] for (topic_id, doc_id) in ids]
    doc_examples = [self.doc_examples[topic_id][doc_id] for (topic_id, doc_id) in ids]
    input_features, extra_features = self.convert_examples_to_features(
      {"sent_examples": examples,
      "doc_examples": doc_examples})
    # a workaround for matching the signature of the abstractmethod

    return self.minibatch_class(
      task_name=self.task_name,
      config=self.config,
      size=len(ids),
      examples=examples,
      teacher_predictions={},
      input_features=input_features,
      extra_features=extra_features)

  def convert_examples_to_features(self, examples):
    sent_examples = examples.pop("sent_examples")
    doc_examples = examples.pop("doc_examples")
    features = []
    extra_features = []
    index = 0
    max_token_length = 0
    max_full_token_length = 0
    max_selected_indices_length = 0

    # The node id in edge should match in keyword id in the document
    # We also need to process document level variables

    for doc_example, sents in zip(doc_examples, sent_examples):
      extra_features.append(
        DocPairwiseFeature(
          label_ids=doc_example.label_ids,
          edge_data=doc_example.edge_data,
          doc_span=(index, index + len(sents))))
      index += len(sents)
      for example in sents:
        max_token_length = max(max_token_length, example.len)
        max_full_token_length = max(max_full_token_length, len(example.text_b_full_token_spans))
        max_selected_indices_length = max(max_selected_indices_length, len(example.selected_indices))

    for doc in sent_examples:
      for example in doc:
        padding = [0] * (max_token_length - example.len)
        input_ids = example.input_ids + padding
        input_mask = example.input_mask + padding
        segment_ids = example.segment_ids + padding
        is_head = example.is_head + padding
        token_spans = example.text_b_full_token_spans + [(1, 0)] * (max_full_token_length - len(example.text_b_full_token_spans))
        selected_indices = example.selected_indices + [-1] * (max_selected_indices_length - len(example.selected_indices))
        features.append(
          PointwiseFeature(input_ids=input_ids,
                           input_mask=input_mask,
                           segment_ids=segment_ids,
                           is_head=is_head,
                           token_spans=token_spans,
                           selected_indices=selected_indices))

    return features, extra_features

  def decode_to_labels(self, preds, mbs):
    return preds

  @property
  def size(self):
    return self._size





