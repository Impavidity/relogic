"""
This module contains the basic components of dataflow,
including Example, Feature, MiniBatch and DataFlow classes.

"""

import abc
import collections
import json
import random
from typing import Dict

import numpy as np

from relogic.logickit.base import utils


class Example(object, metaclass=abc.ABCMeta):
  """Basic Example class."""

  def __init__(self):
    pass

  @abc.abstractmethod
  def process(self, tokenizer, *inputs, **kwargs):
    """Process function abstract. Need to be implemented in each Subclass"""
    raise NotImplementedError()


class Feature(object, metaclass=abc.ABCMeta):
  """Basic Feature class."""

  def __init__(self):
    pass


class MiniBatch(object, metaclass=abc.ABCMeta):
  """Basic MiniBatch class.

  For minimal requirements, this class needs to have the following arguments.

  Args:
    config : The configuration of the DataFlow to which it belongs.
    examples (List[Example]): A list of examples.
    size (int): The size of the batch.
    input_features (List[Feature]): A list of features (each of them are padded).
    teacher_predictions (Dict): This is for semi-supervised learning.

  """

  def __init__(self, *inputs, **kwargs):
    self.task_name = kwargs.pop("task_name")
    self.config = kwargs.pop("config")
    self.examples = kwargs.pop("examples")
    self.size = kwargs.pop("size")
    self.input_features = kwargs.pop("input_features")
    self.teacher_predictions = kwargs.pop("teacher_predictions")
    if hasattr(self.config, "tasks") and "loss_weight" in self.config.tasks[self.task_name]:
      self.loss_weight = self.config.tasks[self.task_name]["loss_weight"]
    else:
      self.loss_weight = 1

  @abc.abstractmethod
  def generate_input(self, device, use_label):
    """Convert the features to tensors

    Args:
      device (torch.Device): gpu device or cpu.
      use_label (bool): To create the label tensor or not.

    """
    raise NotImplementedError()


def get_bucket(config, length):
  """classify each example into given buckets with its length."""

  for i, (start, end) in enumerate(config.buckets):
    if start <= length < end:
      return config.buckets[i]

  return config.buckets[-1]


class DataFlow(object, metaclass=abc.ABCMeta):
  """DataFlow controls the data process and batch generation.

  The DataFlow adopts examples from Structure or json object.

  Note: Most current implementation is based on BERT model.

  Args:
    config (SimpleNamespace): Configuration for the DataFlow class.
    tokenizer: Tokenizer for string tokenization.

  """

  def __init__(self, config, task_name, tokenizers: Dict, label_mapping):
    self.config = config
    self.task_name = task_name
    self.tokenizers = tokenizers
    self.examples = []
    self.label_mapping = label_mapping

    # if label_mapping_path == "none":
    #   self.label_mapping = {}
    # else:
    #   self.label_mapping = json.load(open(label_mapping_path))

  @abc.abstractmethod
  def process_example(self, example):
    """Basic method for example processing. This method needs be implemented
    case by case. For different Subclass, it has different arguments during
    the example processing.

    """
    raise NotImplementedError()

  def update_with_structures(self, structures):
    """Convert the Structure into Example.

    This method is used during the deployment.

    Args:
      structures (List[Structure]): List of Structure.

    """
    self.examples = [
        self.example_class.from_structure(structure)
        for structure in structures
    ]
    from tqdm import tqdm
    for example in tqdm(self.examples):
      self.process_example(example)

  def update_with_jsons(self, examples):
    """Convert json object into Example.

    This method can be used in deployment or training.

    Args:
      examples: (List[Dict]): List of json objects.

    """
    self.examples = [
        self.example_class.from_json(example) for example in examples
    ]
    for example in self.examples:
      self.process_example(example)

  def update_with_file(self, file_name):
    """Read json objects from file.

    Args:
      file_name (str): Filename.

    """
    examples = []
    with open(file_name) as fin:
      for line in fin:
        examples.append(json.loads(line))
    self.update_with_jsons(examples)

  def endless_minibatches(self, minibatch_size, sequential=False, bucket=True):
    """Generate endless minibatches with given batch size."""

    if sequential:
      print("Use sequential dataset for {}".format(self.task_name))
    elif not bucket:
      print("Use random shuffle dataset for {}".format(self.task_name))
    else:
      print("Use bucket dataset for {}".format(self.task_name))
    while True:
      for minibatch in self.get_minibatches(minibatch_size, sequential=sequential, bucket=bucket):
        yield minibatch

  def get_minibatches(self, minibatch_size, sequential, bucket):
    """Generate list of batch size based on examples.

    There are two modes for generating batches. One is sequential,
    which follows the original example sequence in the dataset.
    The other mode is based on bucketing, to save the memory consumption.

    Args:
      minibatch_size (int): Batch size.
      sequential (bool): To be sequential mode or not.

    """
    if sequential:
      index = 0
      while index < self.size:
        yield self._make_minibatch(
            np.array(range(index, min(index + minibatch_size, self.size))))
        index += minibatch_size
    elif not bucket:
      indices = list(range(self.size))
      random.shuffle(indices)
      indices = np.array(indices)
      index = 0
      while index < self.size:
        yield self._make_minibatch(
            indices[index: min(index + minibatch_size, self.size)])
        index += minibatch_size
    else:
      by_bucket = collections.defaultdict(list)
      for i, example in enumerate(self.examples):
        by_bucket[get_bucket(self.config, example.len)].append(i)
      # save memory by weighting examples so longer sentences
      #   have smaller minibatches
      weight = lambda ind: np.sqrt(self.examples[ind].len)
      total_weight = float(sum(weight(i) for i in range(self.size)))
      weight_per_batch = minibatch_size * total_weight / self.size
      cumulative_weight = 0.0
      id_batches = []
      for _, ids in by_bucket.items():
        ids = np.array(ids)
        np.random.shuffle(ids)
        curr_batch, curr_weight = [], 0.0
        for i, curr_id in enumerate(ids):
          curr_batch.append(curr_id)
          curr_weight += weight(curr_id)
          if (i == len(ids) - 1 or cumulative_weight + curr_weight >=
              (len(id_batches) + 1) * weight_per_batch):
            cumulative_weight += curr_weight
            id_batches.append(np.array(curr_batch))
            curr_batch, curr_weight = [], 0.0
      random.shuffle(id_batches)
      utils.log("Data Flow {}, There are {} batches".format(
          self.__class__, len(id_batches)))
      for id_batch in id_batches:
        yield self._make_minibatch(id_batch)

  def _make_minibatch(self, ids):
    """Make a MiniBatch given ids.

    Given ids, the method chooses the corresponding examples.

    Args:
      ids (List(int)): id list.

    Return:
      MiniBatch: The created Minibatch.
    """
    examples = [self.examples[i] for i in ids]
    input_features = self.convert_examples_to_features(examples=examples)

    return self.minibatch_class(
        task_name=self.task_name,
        config=self.config,
        size=ids.size,
        examples=examples,
        teacher_predictions={},
        input_features=input_features)

  @abc.abstractmethod
  def convert_examples_to_features(self, examples):
    """Basic method abstraction for converting examples to features."""

    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def example_class(self):
    """Return the Example class based on the Subclass."""

    raise NotImplementedError()

  @property
  def minibatch_class(self):
    """Return the MiniBatch class based on the Subclass."""

    raise NotImplementedError()

  @property
  def size(self):
    """The size of the dataset."""

    return len(self.examples)
