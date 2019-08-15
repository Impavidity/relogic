import os
import json
from relogic.logickit.base.constants import NEVER_SPLIT

def configure(config):
  config.buckets = [(0, 15), (15, 40), (40, config.max_seq_length)]
  if config.task_names:
    config.task_names = config.task_names.split(',')
    config.raw_data_path = config.raw_data_path.split(',')
    config.label_mapping_path = config.label_mapping_path.split(',')
    config.train_file = config.train_file.split(',')
    config.dev_file = config.dev_file.split(',')
    config.test_file = config.test_file.split(',')
    assert len(config.task_names) == len(config.raw_data_path) == len(config.label_mapping_path)
    config.tasks = {}
    for task, raw_data_path, label_mapping_path, train_file, dev_file, test_file in zip(
          config.task_names, config.raw_data_path, config.label_mapping_path,
          config.train_file, config.dev_file, config.test_file):
      config.tasks[task] = {}
      config.tasks[task]["raw_data_path"] = raw_data_path
      config.tasks[task]["label_mapping_path"] = label_mapping_path
      config.tasks[task]["train_file"] = train_file
      config.tasks[task]["dev_file"] = dev_file
      config.tasks[task]["test_file"] = test_file
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

  if config.task_names:
    if "rel_extraction" in config.task_names:
      if config.bert_model not in ["bert-base-cased", "bert-large-cased"]:
        raise ValueError("For relation extraction on tacred, the vocab only support bert-base-cased for masking")
      config.vocab_path = "relogic/logickit/vocabs/tacred-{}-vocab.txt".format(config.bert_model)

  config.external_vocab_size = 999996 # a quick patch
  config.external_vocab_embed_size = 300
