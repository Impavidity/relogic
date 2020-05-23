import os
import json
from relogic.logickit.base import utils
from relogic.logickit.dataflow import DataFlow, TASK_TO_DATAFLOW_CLASS_MAP


class LabeledDataLoader(object):
  def __init__(self, config, name, tokenizer):
    self.config = config
    self.task_name = name
    self.raw_data_path = config.tasks[name]["raw_data_path"]
    self.label_mapping_path = config.tasks[name]["label_mapping_path"]
    self.file_names = {
      "train": config.tasks[name]["train_file"],
      "dev": config.tasks[name]["dev_file"],
      "test": config.tasks[name]["test_file"]
    }
    self.tokenizer = tokenizer
    # TODO: these code can not support multi-label set dataset.
    # Will fix this in the future.
    if self.label_mapping:
      self.n_classes = len(self.label_mapping.values())
    else:
      self.n_classes = None
    self.extra_args = {}

  def get_dataset(self, split):
    if (split == 'train'
          and (os.path.exists(os.path.join(self.raw_data_path, "train_subset.txt")) or
               os.path.exists(os.path.join(self.raw_data_path, "train_subset.json")))):
      split = 'train_subset'
    dataflow: DataFlow = self.get_dataflow()
    file_path = os.path.join(self.raw_data_path, self.file_names[split])
    dataflow.update_with_file(file_path)
    self.dataflow = dataflow
    return dataflow

  def get_dataflow(self) -> DataFlow:
    return TASK_TO_DATAFLOW_CLASS_MAP[self.task_name](
      config=self.config, task_name=self.task_name ,tokenizers=self.tokenizer, label_mapping=self.label_mapping)

  @property
  def label_mapping(self):
    if self.label_mapping_path == "none":
      return {}
    if self.label_mapping_path.endswith(".pkl"):
      return utils.load_pickle(self.label_mapping_path)
    elif self.label_mapping_path.endswith(".json"):
      return json.load(open(self.label_mapping_path))
    else:
      raise ValueError(self.label_mapping_path)