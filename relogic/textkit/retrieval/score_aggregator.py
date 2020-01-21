import os
from typing import List
from multiprocessing import Pool
import uuid
import subprocess
from copy import deepcopy
import numpy as np
import argparse

def trec_format_writer(qid2doc_score, output_file_path):
  fout = open(output_file_path, 'w')
  for qid in qid2doc_score:
    for idx, (docid, score) in enumerate(qid2doc_score[qid]):
      fout.write("{} Q0 {} {} {} rerenk\n".format(qid, docid, idx + 1, score))
  fout.close()

def trec_eval_exec(run_file_path, qrels_file_path):
  dir = os.path.abspath(os.path.dirname(__file__))
  trec_eval_path = os.path.join(dir, '../../../evals/trec_eval/trec_eval.9.0.4/trec_eval')

  trec_out = subprocess.check_output([trec_eval_path, qrels_file_path, run_file_path])
  trec_out_lines = str(trec_out, 'utf-8').split('\n')
  mean_average_precision = float(trec_out_lines[5].split('\t')[-1])
  return mean_average_precision

def evaluation(qid2doc_score, qrels_file_path):
  temp_file_name = str(uuid.uuid4())
  trec_format_writer(qid2doc_score, temp_file_name)
  map_value = trec_eval_exec(temp_file_name, qrels_file_path)
  os.remove(temp_file_name)
  return map_value

def document_scorer_parameterized(
    qid2doc_score_lists, params, test_topics):
  qid2doc_score = {}
  print("Evaluating the configuation: ", params)
  for qid in test_topics:
    qid2doc_score[qid] = []
    doc_ids = list(qid2doc_score_lists[0][qid].keys())
    for docid in doc_ids:
      score = 0
      for qid2doc_score_list, weight in zip(qid2doc_score_lists, params):
        score += qid2doc_score_list[qid][docid] * weight
      qid2doc_score[qid].append((docid, score))
  return qid2doc_score

def document_parameterized_scorer_with_result(
      qid2doc_score_lists, params, test_topics, qrels_file_path):
  qid2doc_score = document_scorer_parameterized(qid2doc_score_lists, params, test_topics)
  score = evaluation(qid2doc_score, qrels_file_path)
  return score


class ScoreAggregator:
  def __init__(self):
    pass

  def folds_reader(self, fold_path, k_fold):
    folds = []
    for i in range(k_fold):
      with open(os.path.join(fold_path, "fold_topics_{}.txt".format(i + 1))) as fin:
        folds.append([line.strip() for line in fin])
    return folds

  def doc_doc_combination(
        self,
        sources: List[str],
        fold_path: str,
        k_fold: int,
        output_file_path: str,
        qrels_file_path: str):
    qid2doc_score_lists = [self.trec_format_reader(path) for path in sources]
    folds = self.folds_reader(fold_path, k_fold)
    aggregation = {}
    for i in range(k_fold):
      print("Tuning")
      tuned_topic_ids = []
      for j in range(k_fold):
        tuned_topic_ids.extend(folds[j])
      params, _ = self.doc_doc_parameter_tuning(
        qid2doc_score_lists,
        tuned_topic_ids,
        qrels_file_path
      )
      print("Apply to fold {}".format(i+1))
      qid2doc_score = document_scorer_parameterized(
        qid2doc_score_lists=qid2doc_score_lists,
        params=params,
        test_topics=folds[i])
      aggregation.update(qid2doc_score)
    trec_format_writer(aggregation, output_file_path)
    trec_format_writer(aggregation, output_file_path)

  def process_range(self, r):
    if r == "1-x":
      return "1-x"
    if "-" in r and ":" in r:
      # we assume it is format "t1-t2:step"
      r1, r2 = r.split(":")
      start, end = map(float, r1.split("-"))
      step = float(r2)
      return start, end, step

  def params_generation(self, num, ranges):
    combs = []
    ranges = [self.process_range(r) for r in ranges]
    def dfs(idx, comb):
      if idx == num:
        combs.append(deepcopy(comb))
        return
      else:
        if isinstance(ranges[idx], str):
          comb[idx] = 1 - sum(comb[:idx])
          dfs(idx+1, comb)
        else:
          start, end, step = ranges[idx]
          for weight in np.arange(start, end, step):
            comb[idx] = weight
            dfs(idx+1, comb)
    dfs(0, [0] * num)
    print(sorted(combs))
    return combs

  def doc_doc_parameter_tuning(self,
      qid2doc_score_lists: List, topics: List[str], qrels_file_path: str):
    params_list = self.params_generation(
      num=2,
      ranges=["0-1:0.01", "1-x"])
    p = Pool(64)
    argument_list = []
    for params in params_list:
      argument_list.append((qid2doc_score_lists, params, topics, qrels_file_path))
    scores = p.starmap(document_parameterized_scorer_with_result, argument_list)
    best_score = 0
    best_params = None
    for score, params in zip(scores, params_list):
      if score > best_score:
        best_params = params
        best_score = score
    return best_params, best_score

  def trec_format_reader(self, path):
    qid2doc_score = {}
    with open(path) as fin:
      for line in fin:
        qid, _, docid, _, score, _ = line.strip().split()
        if qid not in qid2doc_score:
          qid2doc_score[qid] = {}
        qid2doc_score[qid][docid] = float(score)
    return qid2doc_score

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--keyword_score_file_path", type=str)
  parser.add_argument("--desc_score_file_path", type=str)
  parser.add_argument("--trec_run_output_path", type=str)
  parser.add_argument("--cross_validation", default=False, action="store_true")
  parser.add_argument("--cv_topics_splits_path", type=str)
  parser.add_argument("--k_fold", default=5, type=int)
  parser.add_argument("--qrels_file_path", type=str)

  args = parser.parse_args()
  score_aggregator = ScoreAggregator()
  score_aggregator.doc_doc_combination(
    sources=[args.keyword_score_file_path, args.desc_score_file_path],
    output_file_path=args.trec_run_output_path,
    fold_path=args.cv_topics_splits_path,
    k_fold=args.k_fold,
    qrels_file_path=args.qrels_file_path)
