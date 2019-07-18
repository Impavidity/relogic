import json


class JointSpanExample(object):
  def __init__(self, guid, text, label):
    self.guid = guid
    self.text = text
    self.label = label
    self.raw_text = text.split()
    self.raw_text_length = len(self.raw_text)

  def process(self, tokenizer, extra_args=None):
    pass

class JointSpanFeature(object):
  def __init__(self, input_ids):
    self.input_ids = input_ids

def get_joint_span_examples(path):
  examples = []
  # Json file
  with open(path) as fin:
    for line in fin:
      example = json.loads(line)
      examples.append(
        JointSpanExample(
          guid=example["guid"],
          text=example["text"],
          label=example["label"]))
  return examples

def convert_joint_span_examples_to_features(examples, max_seq_length, extra_args=None):
  pass



