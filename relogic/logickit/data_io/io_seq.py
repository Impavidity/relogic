import json
from relogic.logickit.utils import indicator_vector, create_tensor
import torch


class SeqExample(object):
  def __init__(self, guid, text, label):
    self.guid = guid
    self.text = text
    self.label = label
    self.raw_text = text.split()
    self.raw_text_length = len(self.raw_text)
    self.label_padding = 'X'
    self.label_padding_id = None
    self.valid = True

  def process(self, tokenizer, extra_args=None):
    self.text_tokens, self.text_is_head = tokenizer.tokenize(self.text)
    self.tokens = ["[CLS]"] + self.text_tokens + ["[SEP]"]
    self.segment_ids = [0] * (len(self.text_tokens) + 2)
    self.is_head = [2] + self.text_is_head + [2]
    self.head_index = [idx for idx, value in enumerate(self.is_head) if value == 1]

    self.input_ids = tokenizer.convert_tokens_to_ids(self.tokens)
    self.input_mask = [1] * len(self.input_ids)

    assert "label_mapping" in extra_args
    label_mapping = extra_args["label_mapping"]
    self.label_padding_id = label_mapping[self.label_padding]

    self.label_ids = [self.label_padding_id] * len(self.input_ids)
    # assert len(self.label) == len(self.head_index), str(len(self.label)) + " " + str(len(self.head_index)) + " ".join(self.label) + \
    #                                                 "\n" + " ".join([str(index) for index in self.head_index]) + \
    #                                                 "\n" + self.text + "\n" + " ".join(self.text_tokens)
    if len(self.label) != len(self.head_index):
      self.valid = False
      return
    for idx, label in zip(self.head_index, self.label):
      self.label_ids[idx] = label_mapping[label]

  @property
  def len(self):
    return len(self.input_ids)


class SeqInputFeature(object):
  def __init__(self, input_ids, input_mask, segment_ids, is_head, label_ids):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.is_head = is_head
    self.label_ids = label_ids


def get_seq_examples(path):
  sentences = []
  if path.endswith("json"):
    with open(path, 'r') as f:
      for line in f:
        example = json.loads(line)
        sentences.append((example["tokens"], example["labels"]))
  else:
    with open(path, 'r') as f:
      sentence = []
      for line in f:
        line = line.strip().split()
        if not line:
          if sentence:
            words, tags = zip(*sentence)
            sentences.append((words, tags))
            sentence = []
          continue
        if line[0] == '-DOCSTART-':
          continue
        if len(line) == 2 and line[0] != '\u200b':
          word, tag = line[0], line[-1]
          sentence.append((word, tag))
  examples = [SeqExample(
    guid=str(idx),
    text=" ".join(list(sentence[0])),
    label=list(sentence[1])) for idx, sentence in enumerate(sentences)]
  return examples

def convert_seq_examples_to_features(examples, max_seq_length, extra_args=None):
  features = []
  max_length = max([example.len for example in examples])
  if max_length > max_seq_length:
    raise ValueError("For Seq task, we do not want to truncate. "
                     "The sequence length {} is larger than max_seq_length {}".format(max_length, max_seq_length))
  for idx, example in enumerate(examples):
    if not example.valid:
      continue
    padding = [0] * (max_length - example.len)
    input_ids = example.input_ids + padding
    input_mask = example.input_mask + padding
    segment_ids = example.segment_ids + padding
    is_head = example.is_head + [2] * (max_length - example.len)
    label_ids = example.label_ids + [example.label_padding_id] * (max_length - example.len)

    features.append(
      SeqInputFeature(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        is_head=is_head,
        label_ids=label_ids))
  return features

def generate_seq_input(mb, config, device, use_label):
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
  inputs["extra_args"] = extra_args
  return inputs