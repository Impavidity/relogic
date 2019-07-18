from relogic.logickit.base.utils import log

import json

class RelExtractionOnePassExample(object):
  def __init__(self, guid, text,
        subj_text, obj_text, subj_span, obj_span, subj_type, obj_type, label, mask=True):
    self.guid = guid
    self.text = text
    self.subj_text = subj_text
    self.obj_text = obj_text
    self.subj_span = subj_span
    self.obj_span = obj_span
    self.sub_type = subj_type
    self.obj_type = obj_type
    self.label = label
    self.raw_text = text.split()
    self.raw_text_length = len(self.raw_text)
  
  def process(self, tokenizer, extra_args=None):
    assert "entity_surface_aware" in extra_args
    entity_surface_aware = extra_args["entity_surface_aware"]

    self.text_tokens, self.text_is_head = tokenizer.tokenize(self.text)
    if entity_surface_aware:
      self.subj_tokens, self.subj_is_head = tokenizer.tokenize(self.subj_text)
      self.obj_tokens, self.obj_is_head = tokenizer.tokenize(self.obj_text)
    
    self.tokens = ["[CLS]"] + self.text_tokens + ["SEP"]
    self.segment_ids = [0] + (len(self.text_tokens) + 2)
    self.is_head = [2] + self.text_is_head + [2]
    self.head_index = [idx for idx, value in enumerate(self.is_head) if value == 1] + [ len(self.is_head) - 1]

    self.input_ids = tokenizer.convert_tokens_to_ids(self.tokens)
    self.input_mask = [1] * len(self.input_ids)

    assert "label_mapping" in extra_args
    label_mapping = extra_args["label_mapping"]

    self.label_ids = [label_mapping[relation] for relation in self.label]

    @property
    def len(self):
      return len(self.input_ids)

class RelExtractionOnePassInputFeature(object):
  def __init__(self, input_ids, input_mask, segment_ids, is_head, label_ids):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.is_head = is_head
    self.label_ids = label_ids
  

def get_relextraction_examples_from_json(path):
  examples = []
  log("Read data from {}".format(path))
  with open(path, 'r') as f:
    for idx, line in enumerate(f):
      example = json.loads(line)
      examples.append(RelExtractionOnePassExample(
        guid = idx,
        text = example["masked_sentence"],
        subj_span = example["subj_index"],
        obj_span = example["obj_index"],
        subj_text = example["subj_text"],
        obj_text = example["obj_text"],
        subj_type = example["subj_type"],
        obj_type = example["obj_type"],
        label = example["relation"]))

  return examples

def get_relextraction_onepass_examples(path):
  return get_relextraction_onepass_examples(path)

def convert_relextractiononepass_examples_to_features(examples, max_seq_length, extra_args=None):
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
    features.append(
      RelExtractionOnePassInputFeature(
        input_ids = input_ids,
        input_mask = input_mask,
        segment_ids = segment_ids,
        is_head = is_head,
        label_ids = example.label_ids))
  return features