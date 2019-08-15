from relogic.logickit.utils import get_span_labels

import argparse
from itertools import groupby
import json

from tqdm import tqdm
import spacy
from spacy.tokens import Doc

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

def main(json_file, dump_file):
  examples = []
  with open(json_file) as fin:
    for line in fin:
      example = json.loads(line)
      example["key"] = " ".join(example["tokens"])
      examples.append(example)

  agg_examples = segmentation(examples)
  with open(dump_file, 'w') as fout:
    for example in agg_examples:
      fout.write(json.dumps(example) + "\n")


def segmentation(examples):
  groups = groupby(examples, key=lambda x: x["key"])
  agg_examples = []
  for group in tqdm(groups):
    text = group[0].split()
    segment_label = ["I"] * len(text)
    segment_label[0] = "B"
    sent = nlp(" ".join(text))
    for idx, t in enumerate(sent):
      if t.pos_ == "ADP" or t.pos_ == "PUNCT":
        segment_label[idx] = 'B'
        if idx + 1 < len(text):
          segment_label[idx + 1] = 'B'
    for sub_item in group[1]:
      spans, _ = get_span_labels(sub_item["labels"], ignore_label=[])
      for span in spans:
        segment_label[span[0]] = 'B'
        if len(text) > span[1] + 1:
          segment_label[span[1] + 1] = 'B'

    example = {}
    example["tokens"] = text
    example["segment_labels"] = segment_label
    agg_examples.append(example)
  return agg_examples


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--json_file")
  parser.add_argument("--dump_file")
  args = parser.parse_args()
  main(args.json_file, args.dump_file)