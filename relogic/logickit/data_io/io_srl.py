from relogic.logickit.utils import indicator_vector

class SRLExample(object):
  def __init__(self, guid, text,
        predicate_text, predicate_index, label, predicate_window=0):
    self.guid = guid
    self.text = text
    self.predicate_index = predicate_index
    self.predicate_text = predicate_text
    self.label = label
    self.raw_text = text.split()
    self.raw_text_length = len(self.raw_text)
    self.label_padding = 'X'
    self.label_padding_id = None
    self.predicate_window = predicate_window
    if predicate_window > 0:
      self.predicate_text = self.expand_predicate(self.raw_text, self.predicate_index, predicate_window)

  def expand_predicate(self, text, index, predicate_window):
    span = []
    for i in range(index-predicate_window, index+predicate_window+1):
      if i > 0 and i < len(text):
        span.append(text[i])
      else:
        span.append('[MASK]')
    return " ".join(span)

  def process(self, tokenizer, extra_args=None):
    assert "predicate_surface_aware" in extra_args
    predicate_surface_aware = extra_args["predicate_surface_aware"]

    self.text_tokens, self.text_is_head = tokenizer.tokenize(self.text)
    if predicate_surface_aware:
      self.predicate_tokens, self.predicate_is_head = tokenizer.tokenize(self.predicate_text)

    self.tokens = ["[CLS]"] + self.text_tokens + ["[SEP]"]
    self.segment_ids = [0] * (len(self.text_tokens) + 2)
    self.is_head = [2] + self.text_is_head + [2]

    self.head_index = [idx for idx, value in enumerate(self.is_head) if value == 1] + [ len(self.is_head) - 1]
    # Head index only count for sentence, not for predicate surface form.

    if predicate_surface_aware:
      self.tokens = self.tokens + self.predicate_tokens + ["[SEP]"]
      self.segment_ids = self.segment_ids + [1] * (len(self.predicate_tokens) + 1)
      self.is_head = self.is_head + self.predicate_is_head + [2]

    self.input_ids = tokenizer.convert_tokens_to_ids(self.tokens)
    self.input_mask = [1] * len(self.input_ids)

    if "use_span_annotation":
      span = list(range(self.head_index[self.predicate_index], self.head_index[self.predicate_index+1]))
      self.is_predicate = indicator_vector(
        index=span,
        length=len(self.input_ids),
        default_label=0,
        indicator_label=1)
    else:
      self.is_predicate = indicator_vector(
        index=[self.predicate_index],
        length=len(self.input_ids),
        head_index=self.head_index,
        default_label=0,
        indicator_label=1)

    assert "label_mapping" in extra_args
    label_mapping = extra_args["label_mapping"]
    self.label_padding_id = label_mapping[self.label_padding]

    # label mapping: str -> int
    self.label_ids = [self.label_padding_id] * len(self.input_ids)
    assert len(self.label) == (len(self.head_index) - (1 + self.predicate_window * 2))
    # extra token for predicate
    for idx, label in zip(self.head_index[:-(1 + self.predicate_window * 2)], self.label):
      self.label_ids[idx] = label_mapping[label]

    if "use_span_annotation" in extra_args:
      use_span_annotation = extra_args["use_span_annotation"]
      if use_span_annotation:
        prev_label = None
        for idx, ind in zip(range(len(self.label_ids)), self.is_head):
          if ind == 1:
            prev_label = self.label_ids[idx]
          elif ind == 0:
            self.label_ids[idx] = prev_label

  @property
  def len(self):
    return len(self.input_ids)


class SRLInputFeature(object):
  def __init__(self, input_ids, input_mask, segment_ids, is_head, is_predicate, label_ids):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.is_head = is_head
    self.is_predicate = is_predicate
    self.label_ids = label_ids


def get_srl_examples(path):
  sentences = []
  with open(path, 'r') as f:
    sentence, sent_idx, predicate_text, predicate_index = [], None, None, None
    for line in f:
      line = line.strip().split()
      if not line:
        if sentence:
          words, tags = zip(*sentence)
          sentences.append((sent_idx, words, tags, predicate_text, predicate_index))
          sentence, sent_idx, predicate_text, predicate_index = [], None, None, None
        continue
      if len(line) == 2:
        word, tag = line[0], line[1]
        sentence.append((word, tag))
      elif len(line) == 4:
        sent_idx, predicate_text, predicate_index = int(line[0]), line[3], int(line[1])
  examples = [SRLExample(
    guid=sentence[0],
    text=" ".join(list(sentence[1])),
    label=list(sentence[2]),
    predicate_text=sentence[3],
    predicate_index=sentence[4],
    predicate_window=0) for sentence in sentences]
  return examples

def convert_srl_examples_to_features(examples, max_seq_length, extra_args=None):
  features = []
  max_length = max([example.len for example in examples])
  if max_length > max_seq_length:
    raise ValueError("For SRL task, we do not want to truncate. "
                     "The sequence length {} is larger than max_seq_length {}".format(max_length, max_seq_length))

  for idx, example in enumerate(examples):
    padding = [0] * (max_length - example.len)
    input_ids = example.input_ids + padding
    input_mask = example.input_mask + padding
    segment_ids = example.segment_ids + padding
    is_head = example.is_head + [2] * (max_length - example.len)
    is_predicate = example.is_predicate + padding
    label_ids = example.label_ids + [example.label_padding_id] * (max_length - example.len)

    features.append(
      SRLInputFeature(
        input_ids = input_ids,
        input_mask = input_mask,
        segment_ids = segment_ids,
        is_head = is_head,
        is_predicate=is_predicate,
        label_ids=label_ids))

  return features


