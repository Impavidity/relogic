import sys
import json
from typing import List, Dict
from relogic.logickit.scripts.squad20_eval import normalize_answer, compute_exact, compute_f1
from itertools import groupby
import argparse
import operator

def calculate_score(gold_answer_list: List, pred_answer_list: List):
  if len(gold_answer_list) == 0:
    if len(pred_answer_list) == 0:
      return (1.0, 1.0, 1.0)
    else:
      return (0.0, 1.0, 0.0)
  elif len(pred_answer_list) == 0:
    return (1.0, 0.0, 0.0)
  else:
    pred_answer_list = list(set([normalize_answer(ans) for ans in pred_answer_list]))
    gold_answer_list = list(set([normalize_answer(ans) for ans in gold_answer_list]))

    c = 0
    for ans in pred_answer_list:
      if ans in gold_answer_list:
        c += 1
    p = c / len(pred_answer_list)
    r = c / len(gold_answer_list)
    if p == 0 or r == 0:
      f1 = 0
    else:
      f1 = 2 * p * r / (p+r)
    return (p, r, f1)

def calculate_score_max_over_ground_truths(gold_answer_list: List, pred_answer_list: List):
  """Given a prediction and multiple valid answers, return the score of
  the best prediction-answer_n pair based on exact match"""
  best_scores_for_ground_truths = (0, 0)

  for ground_truth in gold_answer_list:
    for predicted_ans in pred_answer_list:
      em = compute_exact(ground_truth, predicted_ans)
      f1 = compute_f1(ground_truth, predicted_ans)
      if best_scores_for_ground_truths[0] < em:
        best_scores_for_ground_truths = (em, f1)
  return best_scores_for_ground_truths

def rc_evalutate(gold_data: List, pred_answers: List):
  if len(gold_data) != len(pred_answers):
    print("Gold data length: {} != Pred answers length: {}".format(
      len(gold_data), len(pred_answers)))
  total, f1_sum, em_sum = 0, 0, 0
  dump_results = []
  for entry, prediction in zip(gold_data, pred_answers):
    if (len(entry["Parses"]) == 0):
      print("Enpty Parses in gold set. Exist!")
      exit()
    total += 1

    qid, prediction = prediction[0], prediction[1] # (id, list of answers)
    all_parse_answers = []
    for parse in entry["Parses"]:
      parse_answers = [ans["EntityName"] if ans["EntityName"] is not None else ans["AnswerArgument"]
                       for ans in parse["Answers"]]
      all_parse_answers.extend(parse_answers)
    all_parse_answers = list(set(all_parse_answers))
    em, f1 = calculate_score_max_over_ground_truths(all_parse_answers, prediction)
    dump_results.append((qid, prediction, f1, em, all_parse_answers))
    f1_sum += f1
    em_sum += em

  return [("f1", f1_sum / total),
          ("exact_match", em_sum / total)], dump_results

def sp_evaluate(gold_data: List, pred_answers: List):
  if len(gold_data) != len(pred_answers):
    print("Gold data length: {} != Pred answers length: {}".format(
      len(gold_data), len(pred_answers)))
  total, f1_sum, rec_sum, prec_sum, num_correct = 0, 0, 0, 0, 0

  for entry, prediction in zip(gold_data, pred_answers):
    if (len(entry["Parses"]) == 0):
      print("Enpty Parses in gold set. Exist!")
      exit()
    total += 1

    best_f1 = -1
    best_rec = -1
    best_prec = -1
    prediction = prediction[1] # (id, list of answers)

    for parse in entry["Parses"]:
      parse_answers = [ans["EntityName"] if ans["EntityName"] is not None else ans["AnswerArgument"]
                       for ans in parse["Answers"]]


      prec, rec, f1 = calculate_score(parse_answers, prediction)
      if f1 > best_f1:
        best_f1 = f1
        best_rec = rec
        best_prec = prec

    f1_sum += best_f1
    rec_sum += best_rec
    prec_sum += best_prec

    if best_f1 == 1.0:
      num_correct += 1
  precision = prec_sum / total
  recall = rec_sum / total
  if precision == 0 or recall == 0:
    f1 = 0
  else:
    f1 = 2 * precision * recall / (precision + recall)
  return [
    ("f1", f1),
    ("P", precision),
    ("R", recall),
    ("exact_match", num_correct / total)]

def aggregate_answers(pred_answers: Dict, span_scores: Dict) -> List:
  groups = groupby(pred_answers.items(), key=lambda x: x[0].split('.')[0])
  prediction = []
  for k, vs in groups:
    # We use confidence score based voting method to aggregate results
    answer_text_score = {}
    for v in vs:
      if len(v[1]) > 0:
        # Filter out empty span
        if v[1] not in answer_text_score:
          answer_text_score[v[1]] = 0
        answer_text_score[v[1]] += span_scores[v[0]]
    if len(answer_text_score) == 0:
      best_ans = ""
    else:
      best_ans = max(answer_text_score, key=answer_text_score.get)
    prediction.append((k, [best_ans]))
  prediction = sorted(prediction, key=lambda x: int(x[0]), reverse=False)
  return prediction

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--gold_file", type=str)
  parser.add_argument("--pred_file", type=str)
  parser.add_argument("--ir_file", type=str)
  parser.add_argument("--span_score_file", type=str)
  parser.add_argument("--dump_file", type=str)

  args = parser.parse_args()

  gold_data = json.load(open(args.gold_file))["Questions"]

  pred_answers = json.load(open(args.pred_file))

  span_score_file = json.load(open(args.span_score_file))
  ir_score_file = None

  pred_answers = aggregate_answers(pred_answers, span_score_file)

  metrics, dump_results = rc_evalutate(gold_data, pred_answers)
  print(metrics)

  with open(args.dump_file, "w") as fout:
    json.dump(dump_results, fout, indent=2)

