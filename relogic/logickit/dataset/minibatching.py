import collections
import numpy as np
from relogic.logickit.base import utils
import random
from relogic.logickit.data_io import convert_examples_to_features
from relogic.logickit.data_io.io_unlabeled import UnlabeledExample
import os

def get_bucket(config, l):
  for i, (s, e) in enumerate(config.buckets):
    if s <= l < e:
      return config.buckets[i]

class Dataset(object):
  def __init__(self, config, examples, task_name="unlabeled",
               is_training=False, split=None, label_mapping=None, extra_args=None):
    self.config = config
    self.examples = examples
    self.size = len(examples)
    self.task_name = task_name
    self.is_training = is_training
    self.label_mapping = label_mapping
    self.extra_args=extra_args
    self.split = split
    utils.log("Task name: {}, Total {} Examples".format(task_name, self.size))

  def get_minibatches(self, minibatch_size):
    by_bucket = collections.defaultdict(list)
    for i, e in enumerate(self.examples):
      by_bucket[get_bucket(self.config, e.len)].append(i)
    # save memory by weighting examples so longer sentences
    #   have smaller minibatches
    weight = lambda ind: np.sqrt(self.examples[ind].len)
    total_weight = float(sum(weight(i) for i in range(self.size)))
    weight_per_batch = minibatch_size * total_weight / self.size
    cumulative_weight = 0.0
    id_batches = []
    for r, ids in by_bucket.items():
      ids = np.array(ids)
      np.random.shuffle(ids)
      curr_batch, curr_weight = [], 0.0
      for i, curr_id in enumerate(ids):
        curr_batch.append(curr_id)
        curr_weight += weight(curr_id)
        if (i == len(ids) - 1 or
              cumulative_weight + curr_weight >=
              (len(id_batches) + 1) * weight_per_batch):
          cumulative_weight += curr_weight
          id_batches.append(np.array(curr_batch))
          curr_batch, curr_weight = [], 0.0
    random.shuffle(id_batches)
    utils.log("Task name: {}, There are {} batches".format(self.task_name, len(id_batches)))
    for id_batch in id_batches:
      yield self._make_minibatch(id_batch)


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

class UnlabeledDataReader(object):
  def __init__(self, config, starting_file=0, starting_line=0, one_pass=False, tokenizer=None):
    self.config = config
    self.current_file = starting_file
    self.current_line = starting_line
    self.one_pass = one_pass
    self.tokenizer = tokenizer

  def endless_minibatches(self, train_batch_size):
    for examples in self.get_unlabeled_examples():
      d = Dataset(self.config, examples, "unlabeled", is_training=True)
      for mb in d.get_minibatches(train_batch_size):
        yield mb


  def get_unlabeled_examples(self):
    lines = []
    for words in self.get_unlabeled_sentences():
      lines.append(words)
      if len(lines) >= 10000:
        yield self.make_examples(lines)
        lines = []

  def get_unlabeled_sentences(self):
    while True:
      file_ids_and_names = sorted([
        (int(fname.split('-')[1].replace('.txt', '')), fname) for fname in
        os.listdir(self.config.unsupervised_data)])
      for fid, fname in file_ids_and_names:
        if fid < self.current_file:
          continue
        self.current_file = fid
        self.current_line = 0
        with open(os.path.join(self.config.unsupervised_data,
                                             fname), 'r') as f:
          for i, line in enumerate(f):
            if i < self.current_line:
              continue
            self.current_line = i
            words = line.strip().split()
            if len(words) < self.config.max_seq_length:
              yield words
      self.current_file = 0
      self.current_line = 0
      if self.one_pass:
        break

  def make_examples(self, sentences):
    examples = [UnlabeledExample(
      guid="unlabeled",
      text=" ".join(list(sentence))) for sentence in sentences]
    for example in examples:
      example.process(self.tokenizer, self.config.max_seq_length)
    return examples

Minibatch = collections.namedtuple('Minibatch', [
  'task_name', 'size', 'examples', 'ids',
  'teacher_predictions', 'input_features'])
