import argparse
import json

def preprocess(input_file, output_file):
  examples = []
  tokens, labels, arcs, pos = [], [], [], []
  with open(input_file) as fin:
    for line in fin:
      if line == "\n":
        examples.append({
          "tokens": tokens,
          "labels": labels,
          "arcs": arcs,
          "pos": pos
        })
        tokens, labels, arcs, pos = [], [], [], []
      else:
        items = line.strip().split('\t')
        tokens.append(items[1])
        labels.append(items[7].split(":")[0])
        pos.append(items[4])
        arcs.append(int(items[6]))
  with open(output_file, 'w') as fout:
    for example in examples:
      fout.write(json.dumps(example)+"\n")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_file", type=str)
  parser.add_argument("--output_file", type=str)
  args = parser.parse_args()
  preprocess(args.input_file, args.output_file)