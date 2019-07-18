import argparse
import math
def main(input_file, dump_file):
  with open(input_file) as fin:
    sentences_scores = []
    scores = []
    for line in open(input_file):
      if len(line.strip()) == 0:
        sentences_scores.append(scores)
        scores = []
        continue
      token, _, _, confidence = line.strip().split()
      scores.append(float(confidence))
  with open(dump_file, 'w') as fout:
    for idx, scores in enumerate(sentences_scores):
      sum_score = sum([math.log(score) for score in scores]) / len(scores)
      fout.write("{}\t{}\n".format(idx, sum_score))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_file")
  parser.add_argument("--dump_file")
  args = parser.parse_args()
  main(args.input_file, args.dump_file)