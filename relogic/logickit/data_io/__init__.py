import os

import torch

from relogic.logickit.data_io.io_matching import get_matching_examples, convert_matching_examples_to_features
from relogic.logickit.data_io.io_pair import get_pair_examples, convert_pair_examples_to_features
from relogic.logickit.data_io.io_reading_comprehension import get_reading_comprehension_examples, \
  convert_reading_comprehension_examples_to_features
from relogic.logickit.data_io.io_relation import get_relextraction_examples, convert_relextraction_examples_to_features, \
  generate_rel_extraction_input
from relogic.logickit.data_io.io_seq import get_seq_examples, convert_seq_examples_to_features
from relogic.logickit.data_io.io_singleton import get_singleton_examples, convert_singleton_examples_to_features
from relogic.logickit.data_io.io_srl import get_srl_examples, convert_srl_examples_to_features, generate_srl_input
from relogic.logickit.data_io.io_unlabeled import convert_unlabeled_examples_to_features
from relogic.logickit.utils.utils import create_tensor


def get_labeled_examples(split, raw_data_path, task):
  """
  Uniform interface of dataset reader
  :param split: train, dev, test
  :param raw_data_path: path to dataset
  :param task: ner, er, srl
  :return: list of examples
  """
  path = os.path.join(raw_data_path, split + ".txt")
  if not os.path.exists(path):
    path = os.path.join(raw_data_path, split + ".json")
  if not os.path.exists(path):
    raise ValueError("There is no file in {}".format(raw_data_path))
  if task in ["srl", "srl_conll05", "srl_conll09", "srl_conll12"]:
    return get_srl_examples(path)
  elif task in ["er", "ner", "predicate_sense"]:
    return get_seq_examples(path)
  elif task in ["matching"]:
    return get_matching_examples(path)
  elif task in ["rel_extraction"]:
    return get_relextraction_examples(path)
  elif task in ["pair_matching"]:
    if "train" in split:
      return get_pair_examples(path)
    else:
      return get_singleton_examples(path)
  elif task in ["squad11", "squad20"]:
    return get_reading_comprehension_examples(path)
  else:
    raise ValueError("Wrong task name {}".format(task))


def convert_examples_to_features(examples, max_seq_length, task_name, extra_args=None):
  if task_name in ["srl", "srl_conll05", "srl_conll09", "srl_conll12"]:
    return convert_srl_examples_to_features(examples, max_seq_length, extra_args)
  elif task_name in ["er", "ner", "predicate_sense"]:
    return convert_seq_examples_to_features(examples, max_seq_length, extra_args=None)
  elif task_name in ["matching"]:
    return convert_matching_examples_to_features(examples, max_seq_length, extra_args)
  elif task_name in ["rel_extraction"]:
    return convert_relextraction_examples_to_features(examples, max_seq_length, extra_args)
  elif task_name in ["unlabeled"]:
    return convert_unlabeled_examples_to_features(examples, max_seq_length, extra_args)
  elif task_name in ["pair_matching"]:
    assert "is_training" in extra_args
    if extra_args["is_training"]:
      return convert_pair_examples_to_features(examples, max_seq_length, extra_args)
    else:
      return convert_singleton_examples_to_features(examples, max_seq_length, extra_args)
  elif task_name in ["squad11", "squad20"]:
    return convert_reading_comprehension_examples_to_features(examples, max_seq_length, extra_args)
  else:
    raise ValueError("Wrong task name {}".format(task_name))


def generate_input(mb, config, device, use_label=True):
  if mb.task_name in ["span_gcn"]:
    return span_gcn_patching(mb, config, device, use_label)
  if mb.task_name in ["rel_extraction"]:
    return generate_rel_extraction_input(mb, config, device, use_label)
  if mb.task_name in ["srl"]:
    return generate_srl_input(mb, config, device, use_label)
  else:
    return patching(mb, config, device, use_label)


def span_gcn_patching(mb, config, device, use_label):
  inputs = {}
  inputs["task_name"] = mb.task_name
  inputs["input_ids"] = create_tensor(mb.input_features, "input_ids", torch.long, device)
  inputs["input_mask"] = create_tensor(mb.input_features, "input_mask", torch.long, device)
  inputs["segment_ids"] = create_tensor(mb.input_features, "segment_ids", torch.long, device)
  if use_label:
    inputs["label_ids"] = create_tensor(mb.input_features, "label_ids", torch.long, device)
  extra_args = {"selected_non_final_layers": [4]}
  inputs["extra_args"] = extra_args
  return inputs


def patching(mb, config, device, use_label):
  input_ids = torch.tensor([f.input_ids for f in mb.input_features], dtype=torch.long).to(device)
  input_mask = torch.tensor([f.input_mask for f in mb.input_features], dtype=torch.long).to(device)
  if mb.task_name not in ["squad11", "squad20"]:
    input_head = torch.tensor([f.is_head for f in mb.input_features], dtype=torch.long).to(device)
  else:
    input_head = None
  segment_ids = torch.tensor([f.segment_ids for f in mb.input_features], dtype=torch.long).to(device)
  if use_label:
    label_ids = torch.tensor([f.label_ids for f in mb.input_features], dtype=torch.long).to(device)
  else:
    label_ids = None
  extra_args = {}
  if mb.task_name in ["srl", "srl_conll05", "srl_conll09", "srl_conll12"]:
    is_predicate_id = torch.tensor([f.is_predicate for f in mb.input_features], dtype=torch.long).to(device)
    extra_args["is_predicate_id"] = is_predicate_id
  if mb.task_name == 'rel_extraction':
    subj_indicator = torch.tensor([f.subj_indicator for f in mb.input_features], dtype=torch.long).to(device)
    obj_indicator = torch.tensor([f.obj_indicator for f in mb.input_features], dtype=torch.long).to(device)
    extra_args['subj_indicator'] = subj_indicator
    extra_args['obj_indicator'] = obj_indicator
  if mb.task_name == "predicate_sense":
    temp = torch.tensor([f.label_ids for f in mb.input_features], dtype=torch.long).to(device)
    # hard code 'O' == 0 'X' == 22
    extra_args["is_predicate_id"] = (temp != 0) & (temp != 22)
  if config.branching_encoder:
    extra_args["route_path"] = config.task_route_paths[mb.task_name]
  return input_ids, input_mask, input_head, segment_ids, label_ids, extra_args