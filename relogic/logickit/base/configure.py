import os
import json
from relogic.logickit.base.constants import NEVER_SPLIT

def configure(config):
  config.buckets = [(0, 15), (15, 40), (40, config.max_seq_length)]
  if config.task_names:
    config.task_names = config.task_names.split(',')
    config.raw_data_path = config.raw_data_path.split(',')
    config.label_mapping_path = config.label_mapping_path.split(',')
    assert len(config.task_names) == len(config.raw_data_path) == len(config.label_mapping_path)
    config.tasks = {}
    for task, raw_data_path, label_mapping_path in zip(config.task_names, config.raw_data_path, config.label_mapping_path):
      config.tasks[task] = {}
      config.tasks[task]["raw_data_path"] = raw_data_path
      config.tasks[task]["label_mapping_path"] = label_mapping_path
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

  if "rel_extraction" in config.task_names:
    if config.bert_model not in ["bert-base-cased", "bert-large-cased"]:
      raise ValueError("For relation extraction on tacred, the vocab only support bert-base-cased for masking")
    config.vocab_path = "vocabs/tacred-{}-vocab.txt".format(config.bert_model)
