import json


class JointSRLExample(object):
  """
  Joint Predicate Detection and Argument Prediction. Basically the input is a sentence.
  """
  def __init__(self, guid, text, predicate_list=None, argument_list=None):
    self.guid = guid
    self.text = text
    self.raw_text = text.split()
    self.raw_text_length = len(self.raw_text)
  
  def process(self, tokenizer, extra_args=None):
    self.text_tokens, self.text_is_head = tokenizer.tokenize(self.text)

    self.tokens = ["[CLS]"] + self.text_tokens + ["[SEP]"]
    self.segment_ids = [0] * (len(self.text_tokens) + 2)
    self.input_ids = tokenizer.convert_tokens_to_ids(self.tokens)
    self.input_mask = [1] * len(self.input_ids)

  @property
  def len(self):
    return len(self.input_ids)

class JointSRLInputFeature(object):
  def __init__(self, input_ids, input_mask, segment_ids, is_head):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.is_head = is_head

def get_joint_srl_examples(path):
  examples = []
  with open(path, 'r') as f:
    for line in f:
      example = json.loads(line)
      examples.append(
        JointSRLExample(
          guid=example["guid"],
          text=" ".join(example["tokens"])))
  return examples

def convert_joint_srl_examples_to_features(examples, max_seq_length, extra_args=None):
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

    features.append(
      JointSRLInputFeature(
        input_ids = input_ids,
        input_mask = input_mask,
        segment_ids = segment_ids,
        is_head = is_head))
  return features

if __name__ == "__main__":
  from tokenizer.tokenization import BertTokenizer
  tokenizer = BertTokenizer.from_pretrained(
      "vocabs/tacred-bert-base-cased-vocab.txt", do_lower_case=False)
  examples = get_joint_srl_examples("data/raw_data/srl/conll05/train.json")
  for example in examples:
    example.process(tokenizer, extra_args=None)