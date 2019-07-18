from relogic.logickit.utils.utils import gen_position_indicator
from relogic.logickit.base.utils import log
import json

from relogic.logickit.utils.utils import create_tensor
import torch
import numpy as np
from typing import List



class RelExtractionExample(object):
  def __init__(self, guid,
               text, subj_text,
               obj_text, subj_span, obj_span, subj_type, obj_type, label, dependency_tree=None, mask=True):
    self.guid = guid
    self.text = text
    self.subj_text =subj_text
    self.obj_text = obj_text
    self.subj_span = subj_span
    self.obj_span = obj_span
    self.subj_type = subj_type
    self.obj_type = obj_type
    self.label = label
    self.raw_text = text.split()
    self.raw_text_length = len(self.raw_text)
    self.use_dependency_tree = dependency_tree is not None
    self.dependency_tree = dependency_tree

    if mask:
      self.raw_text[self.subj_span[0]: self.subj_span[1]] = ["[" +self.subj_type + "-SUBJ]"] * (self.subj_span[1] - self.subj_span[0])
      self.raw_text[self.obj_span[0]: self.obj_span[1]] = ["[" +self.obj_type + "-OBJ]"] * (self.obj_span[1] - self.obj_span[0])
      self.text = " ".join(self.raw_text)


  def process(self, tokenizer, extra_args=None):
    assert "entity_surface_aware" in extra_args
    entity_surface_aware = extra_args["entity_surface_aware"]

    self.text_tokens, self.text_is_head = tokenizer.tokenize(self.text)

    if entity_surface_aware:
      self.subj_tokens, self.subj_is_head = tokenizer.tokenize(self.subj_text)
      self.obj_tokens, self.obj_is_head = tokenizer.tokenize(self.obj_text)

    self.tokens = ["[CLS]"] + self.text_tokens + ["[SEP]"]
    self.segment_ids = [0] * (len(self.text_tokens) + 2)
    self.is_head = [2] + self.text_is_head + [2]
    self.head_index = [idx for idx, value in enumerate(self.is_head) if value == 1] + [ len(self.is_head) - 1]
    # padding here to fix


    if entity_surface_aware:
      self.tokens = self.tokens + self.subj_tokens + ["[SEP]"] + self.obj_tokens + ["[SEP]"]
      self.segment_ids = self.segment_ids + [1] * (len(self.subj_tokens) + len(self.obj_tokens) + 2)
      self.is_head = self.is_head + self.subj_is_head + [2] + self.obj_is_head + [2]


    self.input_ids = tokenizer.convert_tokens_to_ids(self.tokens)
    self.input_mask = [1] * len(self.input_ids)


    # print(self.head_index)
    # print(self.raw_text)
    # print(self.subj_span[0], self.subj_span[1])
    # print(self.head_index[self.subj_span[0]], self.head_index[self.subj_span[1]])
    self.subj_indicator = gen_position_indicator(
      span=(self.head_index[self.subj_span[0]], self.head_index[self.subj_span[1]]),
      length=len(self.text_tokens) + 2) # include CLS and SEP
    self.start_of_subject = self.head_index[self.subj_span[0]]
    self.obj_indicator = gen_position_indicator(
      span=(self.head_index[self.obj_span[0]], self.head_index[self.obj_span[1]]),
      length=len(self.text_tokens) + 2)
    self.start_of_object = self.head_index[self.obj_span[0]]

    assert "label_mapping" in extra_args
    label_mapping = extra_args["label_mapping"]

    self.label_ids = label_mapping[self.label]

    # now we need to process the binary feature for the dependency tree
    # 1. duplicate
    # 2. extend for furface form if necessary
    if self.dependency_tree:
      self.dependency_tree_feature_map = np.zeros((self.len, self.len))
      for query_idx, query_token in enumerate(self.dependency_tree):
        for key_idx, key_token in enumerate(query_token):
          if key_token == 1:
            for idx in range(self.head_index[key_idx], self.head_index[key_idx+1]):
              self.dependency_tree_feature_map[query_idx][idx] = key_token
    else:
      self.dependency_tree_feature_map = None

  @property
  def len(self):
    return len(self.input_ids)


class RelExtractionInputFeature(object):
  def __init__(self, input_ids, input_mask,
               segment_ids, is_head,
               subj_indicator, obj_indicator, label_ids,
               dependency_tree_feature=None,
               start_of_subject=None,
               start_of_object=None):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.is_head = is_head
    self.subj_indicator = subj_indicator
    self.obj_indicator = obj_indicator
    self.label_ids = label_ids
    self.dependency_tree_feature=dependency_tree_feature
    self.start_of_subject = start_of_subject
    self.start_of_object = start_of_object

def get_relextraction_examples_from_txt(path):
  sentences = []
  log("Read data from {}".format(path))
  with open(path, 'r') as f:
    sentence, relation, subj_text, subj_span, obj_text, obj_span, subj_type, obj_type = [], None, None, None, None, None, None, None
    for line in f:
      line = line.strip().split('\t')
      if line[0] == '':
        if sentence:
          sentences.append((sentence, relation, subj_text, obj_text, subj_span, obj_span, subj_type, obj_type))
          sentence, relation, subj_text, subj_span, obj_text, obj_span, subj_type, obj_type = [], None, None, None, None, None, None, None
        continue
      if line[0] == "#Rel#":
        relation = line[1]
      elif line[0] == "#Subj#":
        subj_span = (int(line[1]), int(line[2]))
        subj_text = line[3]
        subj_type = line[4]
      elif line[0] == "#Obj#":
        obj_span = (int(line[1]), int(line[2]))
        obj_text = line[3]
        obj_type = line[4]
      else:
        sentence.append(line[0])
  examples = [RelExtractionExample(
    guid=idx,
    text=" ".join(list(sentence[0])),
    subj_text=sentence[2],
    obj_text=sentence[3],
    subj_span=sentence[4],
    obj_span=sentence[5],
    subj_type=sentence[6],
    obj_type=sentence[7],
    label=sentence[1]) for idx, sentence in enumerate(sentences)]
  return examples

def get_relextraction_examples_from_json(path):
  examples = []
  with open(path, 'r') as f:
    for idx, line in enumerate(f):
      example = json.loads(line)
      examples.append(RelExtractionExample(
        guid=idx,
        text=example["text"],
        subj_text=example["subj_text"],
        obj_text=example["obj_text"],
        subj_span=example["subj_span"],
        obj_span=example["obj_span"],
        subj_type=example["subj_type"],
        obj_type=example["obj_type"],
        label=example["label"],
        dependency_tree=example.get("dep", None)))
  return examples

def get_relextraction_examples(path):
  if path.endswith(".txt"):
    return get_relextraction_examples_from_txt(path)
  elif path.endswith(".json"):
    return get_relextraction_examples_from_json(path)

def convert_relextraction_examples_to_features(examples: List[RelExtractionExample], max_seq_length, extra_args=None):
  features = []
  max_length = max([example.len for example in examples])
  if max_length > max_seq_length:
    raise ValueError("For Relation Extraction Task, we do not want ot truncate. "
                     "The sequence length {} is larger than max_seq_length {}".format(max_length, max_seq_length))

  for idx, example in enumerate(examples):
    padding = [0] * (max_length - example.len)
    input_ids = example.input_ids + padding
    input_mask = example.input_mask + padding
    segment_ids = example.segment_ids + padding
    is_head = example.is_head + [2] * (max_length - example.len)
    label_ids = example.label_ids

    indicator_padding = [0] * (max_length - len(example.subj_indicator))
    subj_indicator = example.subj_indicator + indicator_padding
    obj_indicator = example.obj_indicator + indicator_padding

    dependency_tree_feature = None
    if example.use_dependency_tree:
      dependency_tree_feature = np.zeros((max_length, max_length))
      dependency_tree_feature[:example.len, :example.len] = example.dependency_tree_feature_map

    features.append(
      RelExtractionInputFeature(
        input_ids = input_ids,
        input_mask = input_mask,
        segment_ids = segment_ids,
        is_head = is_head,
        subj_indicator=subj_indicator,
        obj_indicator=obj_indicator,
        label_ids = label_ids,
        dependency_tree_feature=dependency_tree_feature,
        start_of_subject=example.start_of_subject,
        start_of_object=example.start_of_object))

  return features

def generate_rel_extraction_input(mb, config, device, use_label):
  # We need to follow the current Inference API.
  # Basically current Inference class contains a group of tasks.
  # If there is another group of tasks, such as matching, then we will create
  # another Inference such as PairMatch and define there api
  # current api for inference is
  # forward(self, task_name, input_ids, input_mask, input_head, segment_ids, label_ids, extra_args)
  # so all the extra args will go into dict extra_args
  inputs = {}
  inputs["task_name"] = mb.task_name
  inputs["input_ids"] = create_tensor(mb.input_features, "input_ids", torch.long, device)
  inputs["input_mask"] = create_tensor(mb.input_features, "input_mask", torch.long, device)
  inputs["segment_ids"] = create_tensor(mb.input_features, "segment_ids", torch.long, device)
  inputs["input_head"] = create_tensor(mb.input_features, "is_head", torch.long, device)
  if use_label:
    inputs["label_ids"] = create_tensor(mb.input_features, "label_ids", torch.long, device)
  else:
    inputs["label_ids"] = None
  extra_args = {}
  subj_indicator = torch.tensor([f.subj_indicator for f in mb.input_features], dtype=torch.long).to(device)
  obj_indicator = torch.tensor([f.obj_indicator for f in mb.input_features], dtype=torch.long).to(device)
  extra_args['subj_indicator'] = subj_indicator
  extra_args['obj_indicator'] = obj_indicator
  extra_args['start_of_subject'] = torch.tensor([f.start_of_subject for f in mb.input_features], dtype=torch.long).to(device)
  extra_args['start_of_object'] = torch.tensor([f.start_of_object for f in mb.input_features], dtype=torch.long).to(device)
  if hasattr(config, "use_dependency_feature") and config.use_dependency_feature:
    # check argument exists for compatibility
    dependency_feature = torch.tensor([f.dependency_tree_feature.tolist() for f in mb.input_features], dtype=torch.long).to(device)
    extra_args["token_level_attention_mask"] = [dependency_feature] * 4 +  [None] * 8
  inputs["extra_args"] = extra_args
  return inputs
