import os
import json
from relogic.logickit.base.constants import NEVER_SPLIT

def configure(config):
  config.buckets = [(0, 40), (40, 80), (80, config.max_seq_length)]
  if config.task_names and config.restore_path is None:
    config.task_names = config.task_names.split(',')
    config.raw_data_path = config.raw_data_path.split(',')
    config.label_mapping_path = config.label_mapping_path.split(',')
    config.train_file = config.train_file.split(',')
    config.dev_file = config.dev_file.split(',')
    config.test_file = config.test_file.split(',')
    config.loss_weight = config.loss_weight.split(',')
    config.selected_non_final_layers = config.selected_non_final_layers.split(';')
    config.dataset_type = config.dataset_type.split(',')
    if config.qrels_file_path is not None:
      config.qrels_file_path = config.qrels_file_path.split(',')

    # I assume those tasks do not use qrels will put none as placeholder
    config.train_batch_size = config.train_batch_size.split(',')
    config.test_batch_size = config.test_batch_size.split(',')

    if len(config.train_file) != len(config.task_names):
      config.train_file = [config.train_file[0]] * len(config.task_names)
      config.dev_file = [config.dev_file[0]] * len(config.task_names)
      config.test_file = [config.test_file[0]] * len(config.task_names)
    if len(config.loss_weight) != len(config.task_names):
      config.loss_weight = [config.loss_weight[0]] * len(config.task_names)
    if len(config.selected_non_final_layers) != len(config.task_names):
      config.selected_non_final_layers = [config.selected_non_final_layers[0]] * len(config.task_names)
    if len(config.dataset_type) != len(config.task_names):
      config.dataset_type = [config.dataset_type[0]] * len(config.task_names)

    if len(config.train_batch_size) != len(config.task_names):
      config.train_batch_size = [config.train_batch_size[0]] * len(config.task_names)
    if len(config.test_batch_size) != len(config.task_names):
      config.test_batch_size = [config.test_batch_size[0]] * len(config.task_names)

    assert len(config.task_names) == len(config.raw_data_path) == len(config.label_mapping_path)
    config.tasks = {}
    for (task, raw_data_path, label_mapping_path, train_file, dev_file, test_file,
         loss_weight, selected_non_final_layers, dataset_type, train_batch_size, test_batch_size) in zip(
          config.task_names, config.raw_data_path, config.label_mapping_path,
          config.train_file, config.dev_file, config.test_file, config.loss_weight,
          config.selected_non_final_layers, config.dataset_type,
      config.train_batch_size, config.test_batch_size):
      config.tasks[task] = {}
      config.tasks[task]["raw_data_path"] = raw_data_path
      config.tasks[task]["label_mapping_path"] = label_mapping_path
      config.tasks[task]["train_file"] = train_file
      config.tasks[task]["dev_file"] = dev_file
      config.tasks[task]["test_file"] = test_file
      config.tasks[task]["loss_weight"] = float(loss_weight)
      config.tasks[task]["selected_non_final_layers"] = None if selected_non_final_layers == "none" else [
        int(item) for item in selected_non_final_layers.split(',')]
      config.tasks[task]["dataset_type"] = dataset_type
      config.tasks[task]["train_batch_size"] = int(train_batch_size)
      config.tasks[task]["test_batch_size"] = int(test_batch_size)
    if config.qrels_file_path is not None:
      for (task, qrels_file_path) in zip(config.task_names, config.qrels_file_path):
        config.tasks[task]["qrels_file_path"] = qrels_file_path

  if config.output_dir:
    config.progress = os.path.join(config.output_dir, "progress")
    config.history_file = os.path.join(config.output_dir, "history.pkl")
  if config.partial_view_sources:
    config.partial_view_sources = [int(x) for x in config.partial_view_sources.split(',')]

  config.never_split = NEVER_SPLIT
  if config.branching_encoder:
    routing_config = json.load(open(config.routing_config_file))
    config.task_route_paths = routing_config["task_route_paths"]
    config.branching_structure = routing_config["branching_structure"]
  config.ignore_parameters = list(filter(lambda x:len(x.strip())>0, config.ignore_parameters.split(",")))
  config.vocab_path = config.bert_model
  if config.language_id_file is not None:
    config.language_name2id = json.load(config.language_id_file)
  else:
    config.language_name2id = None
  if config.task_names:
    if "rel_extraction" in config.task_names:
      if config.bert_model not in ["bert-base-cased", "bert-large-cased"]:
        raise ValueError("For relation extraction on tacred, the vocab only support bert-base-cased for masking")
      config.vocab_path = "relogic/logickit/vocabs/tacred-{}-vocab.txt".format(config.bert_model)

  config.external_vocab_size = 999996 # a quick patch
  config.external_vocab_embed_size = 300
  if config.metrics != "":
    config.metrics = config.metrics.split(',')
  else:
    config.metrics = None

def update_configure(restore_config, config):
  # quick fix for batch_size
  for task_name in restore_config.tasks:
    if "train_batch_size" not in restore_config.tasks[task_name]:
      restore_config.tasks[task_name]["train_batch_size"] = restore_config.train_batch_size
      restore_config.tasks[task_name]["test_batch_size"] = restore_config.test_batch_size

  if config.qrels_file_path is not None:
    restore_config.qrels_file_path = config.qrels_file_path
  if config.raw_data_path:
    assert config.task_names is not None
    # If user want to change the raw_data_path, they need to provide the 
    # task_name and the raw_data_path. These two arguments should match 
    # each other.
    config.raw_data_path = config.raw_data_path.split(",")
    task_names = config.task_names.split(",")
    # If user want to change the raw_data_path, then they need to change
    # for all tasks.
    # assert len(config.raw_data_path) == len(config.tasks)
    for name, raw_data_path in zip(task_names, config.raw_data_path):
      restore_config.tasks[name]["raw_data_path"] = raw_data_path
  if config.task_names is not None:

    config.test_file = config.test_file.split(",")
    task_names = config.task_names.split(",")
    if len(config.test_file) == len(task_names):
      print("=========")
      print("We assume you are going to update the test file "
            "because you specify the task_names and test_file at the same time")
      for name, test_file in zip(task_names, config.test_file):
        restore_config.tasks[name]["test_file"] = test_file
    
