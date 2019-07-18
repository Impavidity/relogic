import argparse
import random
import json
import os

random.seed(0)

def read_dataset(input_file):
  sentences = []
  with open(input_file, 'r') as f:
    sentence = []
    for line in f:
      line = line.strip().split()
      if not line:
        if sentence:
          words, tags = zip(*sentence)
          sentences.append(
            {"tokens": words, "labels": tags})
          sentence = []
        continue
      if line[0] == '-DOCSTART-':
        continue
      if len(line) == 2 and line[0] != '\u200b':
        word, tag = line[0], line[-1]
        sentence.append((word, tag))
    if sentence:
      words, tags = zip(*sentence)
      sentences.append(
        {"tokens": words, "labels": tags})
  return sentences

def main(input_file, output_file, ratio):
  dataset = read_dataset(input_file)
  size = int(len(dataset) * ratio)
  if not len(dataset) == size:
    samples = random.sample(dataset, size)
  with open(output_file, 'w') as fout:
    for sample in samples:
      fout.write(json.dumps(sample) + "\n")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_file", type=str)
  parser.add_argument("--output_file", type=str)
  parser.add_argument("--ratio", type=float)
  args = parser.parse_args()
  dir = os.path.dirname(args.output_file)
  if not os.path.exists(dir):
    os.mkdir(dir)
  main(args.input_file, args.output_file, args.ratio)