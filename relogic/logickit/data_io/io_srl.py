from relogic.logickit.utils import indicator_vector, create_tensor
import torch
import json
from relogic.logickit.tokenizer import FasttextTokenizer

fasttext_tokenizer = FasttextTokenizer.from_pretrained("wiki-news-300d-1M")

class SRLExample(object):
  def __init__(self, guid, text,
        predicate_text, predicate_index, label,
        predicate_window=0, span_candidates=None, span_candidates_label=None,
        predicate_descriptions=None, argument_descriptions=None):
    """
      We will keep two sets of annotation here an example. 
      One set of annotation is token level based. Bascially it is 
        BIO annotation scheme.
      The other set of annotation is span level based.
        predicate span, argument spans, and argument label. (These annotations can be
        infered from the token level annotation) Currently we put this functionality to
        scorer.
        And candidate spans, which are pre-detected and dump into the data files.
    """
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
    self.span_candidates = span_candidates
    self.span_candidates_label = span_candidates_label
    self.predicate_descriptions = predicate_descriptions
    self.argument_descriptions = argument_descriptions

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

    self.predicate_span_start_index = self.head_index[self.predicate_index]
    self.predicate_span_end_index = self.head_index[self.predicate_index+1]

    if "use_span_annotation" in extra_args:
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

    if extra_args["srl_module_type"] == "sequence_labeling" or extra_args["srl_module_type"] == "description_aware":
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
    elif extra_args["srl_module_type"] == "span_gcn" :
      if self.span_candidates_label is not None:
        self.label_ids = [label_mapping[label] for label in self.span_candidates_label]
      else:
        self.label_ids = None

    if self.span_candidates is not None:
      self.span_candidates_start_index = [self.head_index[span[0]] for span in self.span_candidates]
      self.span_candidates_end_index = [self.head_index[span[1]+1] for span in self.span_candidates]
    else:
      self.span_candidates_start_index = None
      self.span_candidates_end_index = None

    if self.predicate_descriptions is not None:
      self.predicate_descriptions_tokens = [fasttext_tokenizer.tokenize(text)
                                            for text in self.predicate_descriptions]
      self.predicate_descriptions_ids = [fasttext_tokenizer.convert_tokens_to_ids(tokens)
                                         for tokens in self.predicate_descriptions_tokens]
    else:
      self.predicate_descriptions_tokens = None
      self.predicate_descriptions_ids = None

    if self.argument_descriptions is not None:
      self.argument_descriptions_tokens = [
        [fasttext_tokenizer.tokenize(text) for text in args] for args in self.argument_descriptions]
      self.argument_descriptions_ids = [
        [fasttext_tokenizer.convert_tokens_to_ids(tokens) for tokens in args]
                                        for args in self.argument_descriptions_tokens]
    else:
      self.argument_descriptions_tokens = None
      self.argument_descriptions_ids = None


  @property
  def len(self):
    return len(self.input_ids)


class SRLInputFeature(object):
  def __init__(self, input_ids, input_mask, segment_ids, is_head, is_predicate, label_ids,
               predicate_span_start_index, predicate_span_end_index,
               span_candidates_start_index=None, span_candidates_end_index=None,
               predicate_descriptions_ids=None, argument_descriptions_ids=None):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.is_head = is_head
    self.is_predicate = is_predicate
    self.label_ids = label_ids
    self.predicate_span_start_index = predicate_span_start_index
    self.predicate_span_end_index = predicate_span_end_index
    self.span_candidates_start_index = span_candidates_start_index
    self.span_candidates_end_index = span_candidates_end_index
    self.predicate_descriptions_ids = predicate_descriptions_ids
    self.argument_descriptions_ids = argument_descriptions_ids

def get_srl_examples_from_json(path):
  examples = []
  with open(path, 'r') as f:
    for idx, line in enumerate(f):
      example = json.loads(line)
      examples.append(SRLExample(
        guid=idx,
        text=" ".join(example["tokens"]),
        label=example["labels"],
        predicate_text=example["predicate_text"],
        predicate_index=example["predicate_index"],
        span_candidates=example.get("span_candidates", None),
        span_candidates_label=example.get("span_candidates_label", None),
        predicate_descriptions=example.get("predicate_descriptions", None),
        argument_descriptions=example.get("argument_descriptions", None),
        predicate_window=0))
  return examples

def get_srl_examples_from_txt(path):
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


def get_srl_examples(path):
  if path.endswith(".txt"):
    return get_srl_examples_from_txt(path)
  elif path.endswith(".json"):
    return get_srl_examples_from_json(path)

def convert_srl_examples_to_features(examples, max_seq_length, extra_args=None):
  features = []
  max_length = max([example.len for example in examples])
  if examples[0].label_ids is not None:
    max_label_length = max([len(example.label_ids) for example in examples])
  max_span_candidates_length = 0
  if examples[0].span_candidates_start_index is not None:
    max_span_candidates_length = max([len(example.span_candidates_start_index) for example in examples])
  if max_length > max_seq_length:
    raise ValueError("For SRL task, we do not want to truncate. "
                     "The sequence length {} is larger than max_seq_length {}".format(max_length, max_seq_length))

  if examples[0].predicate_descriptions_ids is not None:
    max_predicate_sense_length = max([len(example.predicate_descriptions_ids) for example in examples])
    max_predicate_description_length = max(
      [max([len(description) for description in example.predicate_descriptions_ids]) for example in examples])


  if examples[0].argument_descriptions_ids is not None:
    max_argument_description_length = 0
    max_argument_sense_length = 0
    label_set_size = 0
    for example in examples:
      if len(example.argument_descriptions_ids) > max_argument_sense_length:
        max_argument_sense_length = len(example.argument_descriptions_ids)
      label_set_size = len(example.argument_descriptions_ids[0])
      for desc_in_roleset in example.argument_descriptions_ids:
        for desc in desc_in_roleset:
          if len(desc) > max_argument_description_length:
            max_argument_description_length = len(desc)


  for idx, example in enumerate(examples):
    padding = [0] * (max_length - example.len)
    input_ids = example.input_ids + padding
    input_mask = example.input_mask + padding
    segment_ids = example.segment_ids + padding
    is_head = example.is_head + [2] * (max_length - example.len)
    is_predicate = example.is_predicate + padding
    if example.label_ids is not None:
      label_ids = example.label_ids + [example.label_padding_id] * (max_label_length - len(example.label_ids))
    else:
      label_ids = None
    if max_span_candidates_length > 0:
      span_candidates_start_index = example.span_candidates_start_index + [-1] * (
            max_span_candidates_length - len(example.span_candidates_start_index))
      span_candidates_end_index = example.span_candidates_end_index + [-1] * (
            max_span_candidates_length - len(example.span_candidates_end_index))
    else:
      span_candidates_start_index = None
      span_candidates_end_index = None

    if example.predicate_descriptions_ids is not None:
      predicate_description_ids = []
      for description in example.predicate_descriptions_ids:
        predicate_description_ids.append(description + [0] * (max_predicate_description_length - len(description)))
      while len(predicate_description_ids) < max_predicate_sense_length:
        predicate_description_ids.append([0] * max_predicate_description_length)
    else:
      predicate_description_ids = None


    if example.argument_descriptions_ids is not None:
      argument_descriptions_ids = []
      for arg_desc_in_roleset in example.argument_descriptions_ids:
        arg_desc_in_roleset_ids = []
        for desc in arg_desc_in_roleset:
          arg_desc_in_roleset_ids.append(desc + [0] * (max_argument_description_length - len(desc)))
        argument_descriptions_ids.append(arg_desc_in_roleset_ids)
      while len(argument_descriptions_ids) < max_argument_sense_length:
        argument_descriptions_ids.append([[0] * max_argument_description_length] * label_set_size)
    else:
      argument_descriptions_ids = None

    features.append(
      SRLInputFeature(
        input_ids = input_ids,
        input_mask = input_mask,
        segment_ids = segment_ids,
        is_head = is_head,
        is_predicate=is_predicate,
        label_ids=label_ids,
        span_candidates_start_index=span_candidates_start_index,
        span_candidates_end_index=span_candidates_end_index,
        predicate_span_start_index=example.predicate_span_start_index,
        predicate_span_end_index=example.predicate_span_end_index,
        predicate_descriptions_ids=predicate_description_ids,
        argument_descriptions_ids=argument_descriptions_ids))

  return features

def generate_srl_input(mb, config, device, use_label):
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
  extra_args["is_predicate_id"] = create_tensor(mb.input_features, "is_predicate", torch.long, device)
  if config.use_span_candidates:
    extra_args["span_candidates"] = (
      create_tensor(mb.input_features, "span_candidates_start_index", torch.long, device),
      create_tensor(mb.input_features, "span_candidates_end_index", torch.long, device)
    )
    extra_args["predicate_span"] = (
      create_tensor(mb.input_features, "predicate_span_start_index", torch.long, device),
      create_tensor(mb.input_features, "predicate_span_end_index", torch.long, device)
    )
  if config.use_description:
    extra_args["predicate_descriptions_ids"] = create_tensor(mb.input_features,
                                "predicate_descriptions_ids", torch.long, device)
    extra_args["argument_descriptions_ids"] = create_tensor(mb.input_features,
                                "argument_descriptions_ids", torch.long, device)
  inputs["extra_args"] = extra_args
  return inputs
