class SingletonExample(object):
  def __init__(self, guid, text, gold_pair=None):
    self.guid = guid
    self.text = text
    self.gold_pair = gold_pair
    self.raw_text_length = len(self.text)

  def process(self, tokenizer, extra_args=None):
    self.text_tokens, self.text_is_head = tokenizer.tokenize(self.text)

    assert "max_seq_length" in extra_args
    max_seq_length = extra_args["max_seq_length"]

    self.text_tokens = self.text_tokens[:max_seq_length]
    self.text_is_head = self.text_is_head[:max_seq_length]

    self.tokens = ["[CLS]"] + self.text_tokens + ["[SEP]"]
    self.segment_ids = [0] * (len(self.text_tokens) + 2)
    self.is_head = [2] + self.text_is_head + [2]

    self.input_ids = tokenizer.convert_tokens_to_ids(self.tokens)
    self.input_mask = [1] * len(self.input_ids)

  @property
  def len(self):
    return len(self.input_ids)

  def __str__(self):
    return str(self.__dict__)

class SingletonInputFeature(object):
  def __init__(self, input_ids, input_mask, segment_ids, is_head):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.is_head = is_head

def get_singleton_examples(path):
  examples = []
  with open(path, 'r') as f:
    for line in f:
      pair_id, text, gold_pair = line.strip().split('\t')
      examples.append(
        SingletonExample(
          guid=pair_id,
          text=text,
          gold_pair=gold_pair))
  return examples

def convert_singleton_examples_to_features(examples, max_seq_length, extra_args=None):
  features = []
  max_length = max([example.len for example in examples])

  for idx, example in enumerate(examples):
    padding = [0] * (max_length - example.len)
    input_ids = example.input_ids + padding
    input_mask = example.input_mask + padding
    segment_ids = example.segment_ids + padding
    is_head = example.is_head + [2] * (max_length - example.len)

    features.append(
      SingletonInputFeature(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        is_head=is_head))

  return features