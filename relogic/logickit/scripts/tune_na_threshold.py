import argparse
import json
from itertools import groupby

def main(answer_dump, answer_score, na_probs, threshold):
  # First group items based on the question
  answer_dump_group = dict(groupby(answer_dump.items(), key=lambda x: x[0].split('.')[0]))
  answer_score_group = dict(groupby(answer_score.items(), key=lambda x: x[0].split('.')[0]))
  na_probs_group = dict(groupby(na_probs.items(), key=lambda x: x[0].split('.')[0]))
  for qid in answer_dump_group:
    answers = answer_dump_group[qid]
    answer_scores = answer_score_group[qid]
    answer_na_probs = na_probs_group[qid]



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--answer_dump_file")
  parser.add_argument("--answer_score_file")
  parser.add_argument("--na_prob_file")

  threshold = 1.0
  args = parser.parse_args()
  with open(args.answer_dump_file) as fin:
    answer_dump = json.load(fin)
  with open(args.answer_score_file) as fin:
    answer_score = json.load(fin)
  with open(args.na_prob_file) as fin:
    na_probs = json.load(fin)
  main(answer_dump, answer_score, na_probs, threshold)