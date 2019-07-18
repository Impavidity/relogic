from relogic.logickit.utils import truncate_seq_pair

class MatchingExample(object):
  def __init__(self, guid, text_a, text_b, label, gold_pair):
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label
    self.gold_pair = gold_pair
    self.raw_text_length = len(self.text_a.split()) + len(self.text_b.split())

  def process(self, tokenizer, extra_args=None):
    self.text_a_tokens, self.text_a_is_head = tokenizer.tokenize(self.text_a)
    self.text_b_toekns, self.text_b_is_head = tokenizer.tokenize(self.text_b)

    assert "max_seq_length" in extra_args
    max_seq_length = extra_args["max_seq_length"]

    truncate_seq_pair(self.text_a_tokens, self.text_b_toekns, max_seq_length - 3)
    truncate_seq_pair(self.text_a_is_head, self.text_b_is_head, max_seq_length - 3)

    self.tokens = ["[CLS]"] + self.text_a_tokens + ["[SEP]"] + self.text_b_toekns + ["[SEP]"]
    self.segment_ids = [0] * (len(self.text_a_tokens) + 2) + [1] * (len(self.text_b_toekns) + 1)
    self.is_head = [2] + self.text_a_is_head + [2] + self.text_b_is_head + [2]

    self.input_ids = tokenizer.convert_tokens_to_ids(self.tokens)
    self.input_mask = [1] * len(self.input_ids)

    assert "label_mapping" in extra_args
    label_mapping = extra_args["label_mapping"]

    self.label_ids = label_mapping[self.label]

  @property
  def len(self):
    return len(self.input_ids)

  def __str__(self):
    return str(self.__dict__)

class MatchingInputFeature(object):
  def __init__(self, input_ids, input_mask, segment_ids, is_head, label_ids):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.is_head = is_head
    self.label_ids = label_ids

def get_matching_examples(path):
  examples = []
  with open(path, 'r') as f:
    for line in f:
      pair_id, text_a, text_b, label, gold_pair = line.strip().split("\t")
      examples.append(
        MatchingExample(
          guid=pair_id,
          text_a=text_a,
          text_b=text_b,
          label=label,
          gold_pair=gold_pair))
  return examples

def convert_matching_examples_to_features(examples, max_seq_length, extra_args=None):
  features = []
  max_length = max([example.len for example in examples])

  for idx, example in enumerate(examples):
    padding = [0] * (max_length - example.len)
    input_ids = example.input_ids + padding
    input_mask = example.input_mask + padding
    segment_ids = example.segment_ids + padding
    is_head = example.is_head + [2] * (max_length - example.len)

    features.append(
      MatchingInputFeature(
        input_ids = input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        is_head=is_head,
        label_ids=example.label_ids))

  return features
