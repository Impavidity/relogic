class UnlabeledExample(object):
  def __init__(self, guid, text):
    self.guid = guid
    self.text = text
    self.raw_text = text.split()
    self.raw_text_length = len(self.raw_text)

  def process(self, tokenizer, max_seq_length):
    self.text_tokens, self.text_is_head = tokenizer.tokenize(self.text)
    if len(self.text_tokens) > max_seq_length - 2:
      self.text_tokens = self.text_tokens[:max_seq_length - 2]
      self.text_is_head = self.text_is_head[:max_seq_length - 2]

    self.tokens = ["[CLS]"] + self.text_tokens + ["[SEP]"]
    self.segment_ids = [0] * (len(self.text_tokens) + 2)
    self.is_head = [2] + self.text_is_head + [2]
    self.head_index = [idx for idx, value in enumerate(self.is_head) if value == 1]

    self.input_ids = tokenizer.convert_tokens_to_ids(self.tokens)
    self.input_mask = [1] * len(self.input_ids)

  @property
  def len(self):
    return len(self.input_ids)

class UnlabeledInputFeature(object):
  def __init__(self, input_ids, input_mask, segment_ids, is_head):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.is_head = is_head


def convert_unlabeled_examples_to_features(examples, max_seq_length, extra_args=None):
  features = []
  max_length = max([example.len for example in examples])
  if max_length > max_seq_length:
    raise ValueError("For Seq task, we do not want to truncate. "
                     "The sequence length {} is larger than max_seq_length {}".format(max_length, max_seq_length))
  for idx, example in enumerate(examples):
    padding = [0] * (max_length - example.len)
    input_ids = example.input_ids + padding
    input_mask = example.input_mask + padding
    segment_ids = example.segment_ids + padding
    is_head = example.is_head + [2] * (max_length - example.len)

    features.append(
      UnlabeledInputFeature(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        is_head=is_head))
  return features

