import argparse
import collections
import random


def main(input_file, output_file, num_neg_per_qurey):
  example_collections = collections.defaultdict(list)
  with open(input_file) as fin:
    for line in fin:
      items = line.strip().split('\t')
      pair_id = items[0].split('-')
      pos_id = pair_id[0]
      example_collections[pos_id].append(line)
  with open(output_file, "w") as fout:
    for key in example_collections:
      samples = random.sample(example_collections[key], min(num_neg_per_qurey, len(example_collections[key])))
      for sample in samples:
        fout.write(sample)

if __name__ == "__main__":
  parser = argparse.ArgumentParser() 
  parser.add_argument("--input_file", type=str)
  parser.add_argument("--num_neg_per_query", type=int, default=15)
  parser.add_argument("--output_file", type=str)
  args = parser.parse_args()
  main(args.input_file, args.output_file, args.num_neg_per_query)