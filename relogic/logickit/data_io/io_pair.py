class PairExample(object):
  def __init__(self, guid, text_a, text_b, text_c, gold_pair):
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.text_c = text_c
    self.gold_pair = gold_pair
    self.raw_text_length = len(self.text_a.split()) + len(self.text_b.split()) + len(self.text_c.split())

  def process(self, tokenizer, extra_args=None):
    self.text_a_tokens, self.text_a_is_head = tokenizer.tokenize(self.text_a)
    self.text_b_tokens, self.text_b_is_head = tokenizer.tokenize(self.text_b)
    self.text_c_tokens, self.text_c_is_head = tokenizer.tokenize(self.text_c)

    assert "max_seq_length" in extra_args
    max_seq_length = extra_args["max_seq_length"]

    # [CLS] sentence1 [SEP]
    self.text_a_tokens = self.text_a_tokens[:max_seq_length]
    self.text_b_tokens = self.text_b_tokens[:max_seq_length]
    self.text_c_tokens = self.text_c_tokens[:max_seq_length]


    self.text_a_is_head = self.text_a_is_head[:max_seq_length]
    self.text_b_is_head = self.text_b_is_head[:max_seq_length]
    self.text_c_is_head = self.text_c_is_head[:max_seq_length]

    self.a_tokens = ["[CLS]"] + self.text_a_tokens + ["[SEP]"]
    self.a_segment_ids = [0] * (len(self.text_a_tokens) + 2)
    self.a_is_head = [2] + self.text_a_is_head + [2]

    self.b_tokens = ["[CLS]"] + self.text_b_tokens + ["[SEP]"]
    self.b_segment_ids = [0] * (len(self.text_b_tokens) + 2)
    self.b_is_head = [2] + self.text_b_is_head + [2]

    self.c_tokens = ["[CLS]"] + self.text_c_tokens + ["[SEP]"]
    self.c_segment_ids = [0] * (len(self.text_c_tokens) + 2)
    self.c_is_head = [2] + self.text_c_is_head + [2]

    self.a_input_ids = tokenizer.convert_tokens_to_ids(self.a_tokens)
    self.b_input_ids = tokenizer.convert_tokens_to_ids(self.b_tokens)
    self.c_input_ids = tokenizer.convert_tokens_to_ids(self.c_tokens)
    self.a_input_mask = [1] * len(self.a_input_ids)
    self.b_input_mask = [1] * len(self.b_input_ids)
    self.c_input_mask = [1] * len(self.c_input_ids)

  @property
  def len_a(self):
    return len(self.a_input_ids)
  @property
  def len_b(self):
    return len(self.b_input_ids)
  @property
  def len_c(self):
    return len(self.c_input_ids)
  @property
  def len(self):
    return len(self.a_input_ids) + len(self.b_input_ids) + len(self.c_input_ids)
    

  def __str__(self):
    return str(self.__dict__)

class PairInputFeature(object):
  def __init__(self, a_input_ids, a_input_mask, a_segment_ids, a_is_head,
                     b_input_ids, b_input_mask, b_segment_ids, b_is_head,
                     c_input_ids, c_input_mask, c_segment_ids, c_is_head):
    self.a_input_ids = a_input_ids
    self.a_input_mask = a_input_mask
    self.a_segment_ids = a_segment_ids
    self.a_is_head = a_is_head
    self.b_input_ids = b_input_ids
    self.b_input_mask = b_input_mask
    self.b_segment_ids = b_segment_ids
    self.b_is_head = b_is_head
    self.c_input_ids = c_input_ids
    self.c_input_mask = c_input_mask
    self.c_segment_ids = c_segment_ids
    self.c_is_head = c_is_head


def get_pair_examples(path):
  examples = []
  with open(path, 'r') as f:
    for line in f:
      pair_id, text_a, text_b, text_c, gold_pair = line.strip().split('\t')
      examples.append(
        PairExample(
          guid=pair_id,
          text_a=text_a,
          text_b=text_b,
          text_c=text_c,
          gold_pair=gold_pair))
  return examples

def convert_pair_examples_to_features(examples, max_seq_length, extra_args=None):
  features = []
  a_max_length = max([example.len_a for example in examples])
  b_max_length = max([example.len_b for example in examples])
  c_max_length = max([example.len_c for example in examples])

  for idx, example in enumerate(examples):
    a_padding = [0] * (a_max_length - example.len_a)
    b_padding = [0] * (b_max_length - example.len_b)
    c_padding = [0] * (c_max_length - example.len_c)

    a_input_ids = example.a_input_ids + a_padding
    b_input_ids = example.b_input_ids + b_padding
    c_input_ids = example.c_input_ids + c_padding

    a_input_mask = example.a_input_mask + a_padding
    b_input_mask = example.b_input_mask + b_padding
    c_input_mask = example.c_input_mask + c_padding

    a_segment_ids = example.a_segment_ids + a_padding
    b_segment_ids = example.b_segment_ids + b_padding
    c_segment_ids = example.c_segment_ids + c_padding

    a_is_head = example.a_is_head + [2] * (a_max_length - example.len_a)
    b_is_head = example.b_is_head + [2] * (b_max_length - example.len_b)
    c_is_head = example.c_is_head + [2] * (c_max_length - example.len_c)

    features.append(
      PairInputFeature(
        a_input_ids = a_input_ids,
        a_input_mask = a_input_mask,
        a_segment_ids = a_segment_ids,
        a_is_head = a_is_head,
        b_input_ids = b_input_ids,
        b_input_mask = b_input_mask,
        b_segment_ids = b_segment_ids,
        b_is_head = b_is_head,
        c_input_ids = c_input_ids,
        c_input_mask = c_input_mask,
        c_segment_ids = c_segment_ids,
        c_is_head = c_is_head))

  return features


