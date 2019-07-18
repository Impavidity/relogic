from relogic.logickit.dataset.minibatching import Dataset, Minibatch
from relogic.logickit.base import utils
from relogic.logickit.data_io import convert_examples_to_features
import numpy as np

class SequentialDataset(object):
  def __init__(self, config, examples, task_name="unlabeled",
               is_training=False, split=None, label_mapping=None, extra_args=None):
    self.config = config
    self.examples = examples
    self.size = len(examples)
    self.task_name = task_name
    self.is_training = is_training
    self.label_mapping = label_mapping
    self.extra_args = extra_args
    self.split = split
    utils.log("Task name: {}, Total {} Examples".format(task_name, self.size))


  def get_minibatches(self, minibatch_size):
    index = 0
    while index < self.size:
      yield self._make_minibatch(np.array(range(index, min(index + minibatch_size, self.size))))
      index += minibatch_size

  def endless_minibatches(self, minibatch_size):
    while True:
      for mb in self.get_minibatches(minibatch_size):
        yield mb

  def _make_minibatch(self, ids):
    examples = [self.examples[i] for i in ids]
    input_features = convert_examples_to_features(
      examples=examples,
      max_seq_length=self.config.max_seq_length,
      task_name=self.task_name,
      extra_args={"is_training": "train" in self.split})

    return Minibatch(
      task_name=self.task_name,
      size=ids.size,
      examples=examples,
      ids=ids,
      teacher_predictions={},
      input_features=input_features)

def get_dataset(dataset_type):
  if dataset_type == "sequential":
    return SequentialDataset
  return Dataset
