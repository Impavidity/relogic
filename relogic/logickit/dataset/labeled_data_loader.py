import os
from relogic.logickit.base import utils
from relogic.logickit.data_io import get_labeled_examples
from relogic.logickit.dataset.dataset import get_dataset


class LabeledDataLoader(object):
  def __init__(self, config, name, tokenizer):
    self.config = config
    self.task_name = name
    self.raw_data_path = config.tasks[name]["raw_data_path"]
    self.label_mapping_path = config.tasks[name]["label_mapping_path"]
    self.tokenizer = tokenizer
    # TODO: these code can not support multi-label set dataset.
    # Will fix this in the future.
    if self.label_mapping:
      self.n_classes = len(set(self.label_mapping.values()))
    else:
      self.n_classes = None
    self.extra_args = {}

  def get_dataset(self, split):
    if (split == 'train'
          and (os.path.exists(os.path.join(self.raw_data_path, "train_subset.txt")) or 
               os.path.exists(os.path.join(self.raw_data_path, "train_subset.json")))):
      split = 'train_subset'
    if "train" in split:
      dataset_type = "bucket"
    else:
      dataset_type = "sequential"
    utils.log("Using {} Dataset".format(dataset_type))
    return get_dataset(dataset_type=dataset_type)(
      config=self.config,
      examples=self.get_examples(split),
      task_name=self.task_name,
      is_training=self.config.mode == 'train',
      split=split,
      label_mapping=self.label_mapping,
      extra_args=self.extra_args)

  def get_examples(self, split):
    examples = get_labeled_examples(split=split, raw_data_path=self.raw_data_path, task=self.task_name)
    extra_args={
        "label_mapping": self.label_mapping,
        "max_seq_length": self.config.max_seq_length}
    if self.task_name in ["squad11", "squad20"]:
      extra_args["max_query_length"] = self.config.max_query_length
      extra_args["doc_stride"] = self.config.doc_stride
    if self.task_name in ["rel_extraction"]:
      extra_args["entity_surface_aware"] = self.config.entity_surface_aware
    for example in examples:
      example.process(tokenizer=self.tokenizer, extra_args=extra_args)
      # max_query_length, doc_stride are especially for reading comprehension
    utils.log("{} max sentence raw text length {}; max sentence token length {}".format(
      split,
      max([example.raw_text_length for example in examples]),
      max([example.len for example in examples])))
    return examples

  @property
  def label_mapping(self):
    if self.label_mapping_path == "none":
      return {}
    return utils.load_pickle(self.label_mapping_path)