import argparse
import json

def main(json_file, output_file):
  fout = open(output_file, 'w')
  with open(json_file) as fin:
    for line in fin:
      example = json.loads(line)
      relation = example["relation"]
      sub = example["sub"]
      obj = example["obj"]
      tokens = example["tokens"]
      fout.write("#Rel#\t{}\n".format(relation))
      fout.write("#Subj#\t{}\t{}\t{}\t{}\n".format(
        sub["start"],
        sub["end"],
        sub["text"],
        sub["entityType"]))
      fout.write("#Obj#\t{}\t{}\t{}\t{}\n".format(
        obj["start"],
        obj["end"],
        obj["text"],
        obj["entityType"]))
      for token in tokens:
        fout.write("{}\n".format(token))
      fout.write("\n")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--json_file", type=str)
  parser.add_argument("--output_file", type=str)
  args = parser.parse_args()
  main(args.json_file, args.output_file)