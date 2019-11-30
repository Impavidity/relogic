from relogic.logickit.scorer.scorer import Scorer
from relogic.logickit.utils.utils import softmax
import torch.nn.functional as F
import torch
from tqdm import tqdm
import os
import subprocess
import json

class RecallScorer(Scorer):
  def __init__(self, label_mapping, topk, correct_label='1', dump_to_file=None):
    super(RecallScorer, self).__init__()
    self.label_mapping = label_mapping
    self._inv_label_mapping = {v: k for k, v in label_mapping.items()}
    self._examples = []
    self._preds = []
    self.topk = topk
    self.correct_label = correct_label
    if dump_to_file:
      self.dump_to_file_path = os.path.join(dump_to_file["output_dir"], dump_to_file["task_name"] + "_dump.json")


  def update(self, mbs, predictions, loss, extra_args):
    super(RecallScorer, self).update(mbs, predictions, loss, extra_args)
    for example, preds in zip(mbs.examples, predictions):
      self._examples.append(example)
      self._preds.append(preds)

  def get_loss(self):
    return 0

  def _get_results(self):
    if self.dump_to_file_path:
      self.dump_to_file_handler = open(self.dump_to_file_path, 'w')

    self._n_hit_left, self._n_hit_right, self._n_total_left, self._n_total_right = 0, 0, 0, 0
    pred_collection = [{}, {}] # forward direction and backward direction
    gold_collection = [{}, {}]
    for example, preds in zip(self._examples, self._preds):
      prob = preds[self.label_mapping[self.correct_label]].item()
      query_id, candidate_id, direction = example.guid.split('-')
      direction = int(direction)
      if direction == 1:
        query_id, candidate_id = candidate_id, query_id
      if self.dump_to_file_handler:
        self.dump_to_file_handler.write("{} {} {} {}\n".format(query_id, candidate_id, direction, prob))
      if query_id not in pred_collection[direction]:
        pred_collection[direction][query_id] = []
      pred_collection[direction][query_id].append((candidate_id, prob))

      # if example.label == self.correct_label:
      #   gold_collection[direction][query_id] = candidate_id
      gold_query_id, gold_candidate_id, gold_direction = example.gold_pair.split('-')
      gold_direction = int(gold_direction)
      if gold_direction == 1:
        gold_query_id, gold_candidate_id = gold_candidate_id, gold_query_id
      gold_collection[gold_direction][gold_query_id] = gold_candidate_id

    if len(pred_collection[0]) != len(gold_collection[0]) or len(pred_collection[1]) != len(gold_collection[1]):
      raise ValueError("The query size in pred collectdion {}|{} is different from gold collection {}|{}".format(
        len(pred_collection[0]), len(pred_collection[1]), len(gold_collection[0]), len(gold_collection[1])))
    for d in range(2):
      for query_id in pred_collection[d]:
        sorted_list = sorted(pred_collection[d][query_id], key=lambda x: x[1], reverse=True)
        candidate_ids = [x[0] for x in sorted_list][:self.topk]
        if gold_collection[d][query_id] in candidate_ids:
          if d == 0:
            self._n_hit_left += 1
          else:
            self._n_hit_right += 1
    self._n_total_left = len(gold_collection[0])
    self._n_total_right = len(gold_collection[1])

    if self.dump_to_file_path:
      self.dump_to_file_handler.close()

    return [("hits_left", self._n_hit_left),
            ("hits_right", self._n_hit_right),
            ("total_left", self._n_total_left),
            ("total_right", self._n_total_right),
            ("recall_left", self._n_hit_left / self._n_total_left),
            ("recall_right", self._n_hit_right / self._n_total_right)]

class CartesianMatchingRecallScorer(Scorer):
  def __init__(self, topk, qrels_file_path, dump_to_file=None):
    super(CartesianMatchingRecallScorer, self).__init__()
    self.topk = topk
    self.qrels_file_path = qrels_file_path
    if dump_to_file:
      self.dump_to_file_path = os.path.join(dump_to_file["output_dir"], dump_to_file["task_name"] + "_dump.json")
      self.dump_to_file_handler = open(self.dump_to_file_path, 'w')
    else:
      print("You need to specify the dump_to_file path for the pair matching task")
      exit()

  def update(self, mbs, reprs, loss, extra_args):
    super(CartesianMatchingRecallScorer, self).update(mbs, reprs, loss, extra_args)
    for example, repr in zip(mbs.examples, reprs):
      self.dump_to_file_handler.write(json.dumps({
        "guid": example.guid,
        "feature": " ".join([str(f) for f in repr.cpu().data.numpy()])}) + "\n")

  def get_loss(self):
    return 0

  def _get_results(self):
    self.dump_to_file_handler.close()
    dir = os.path.abspath(os.path.dirname(__file__))
    recall_eval_path = os.path.join(dir, '..', '..', '..', 'evals', 'pair_matching', 'entity_align_eval.py')
    eval_out = subprocess.check_output(["python", recall_eval_path, "-e", self.dump_to_file_path, "-g", self.qrels_file_path])
    eval_out_lines = str(eval_out, 'utf-8').split('\n')
    results = []
    for line in eval_out_lines:
      if line.startswith("Hits@1:"):
        score = float(line.strip().split(" ")[1].strip("%"))
        results.append(score)

    return [("recall_left", results[0]),
            ("recall_right", results[1])]


# class CartesianMatchingRecallScorer(Scorer):
#   def __init__(self, topk, dump_to_file=None):
#     super(CartesianMatchingRecallScorer, self).__init__()
#     self.topk = topk
#     self.left = {}
#     self.right = {}
#     self.left_to_right_gold = {}
#     self.right_to_left_gold = {}
#     if dump_to_file:
#       self.dump_to_file_path = os.path.join(dump_to_file["output_dir"], dump_to_file["task_name"] + "_dump.json")
#
#
#   def update(self, mbs, reprs, loss, extra_args):
#     super(CartesianMatchingRecallScorer, self).update(mbs, reprs, loss, extra_args)
#     for example, repr in zip(mbs.examples, reprs):
#       left_id, right_id, direc = example.guid.split('-')
#       if direc == "0":
#         self.left[left_id] = repr
#       else:
#         self.right[right_id] = repr
#       left_id, right_id, direc = example.gold_pair.split('-')
#       if direc == "0":
#         self.left_to_right_gold[left_id] = right_id
#       else:
#         self.right_to_left_gold[right_id] = left_id
#
#   def get_loss(self):
#     return 0
#
#   def _get_results(self):
#     if self.dump_to_file_path:
#       self.dump_to_file_handler = open(self.dump_to_file_path, 'w')
#
#     self._n_hit_left, self._n_hit_right, self._n_total_left, self._n_total_right = 0, 0, 0, 0
#     if self.dump_to_file_path:
#       for key in self.left:
#         self.dump_to_file_handler.write("{}\t".format(key) + " ".join([str(f) for f in self.left[key].cpu().data.numpy()]) + '\n')
#       for key in self.right:
#         self.dump_to_file_handler.write("{}\t".format(key) + " ".join([str(f) for f in self.right[key].cpu().data.numpy()]) + '\n')
#     left_to_right_distance = {}
#     right_to_left_distance = {}
#     print("Evaluating left to right")
#     for left_id in self.left.keys():
#       left_to_right_distance[left_id] = {}
#       for right_id in self.right.keys():
#         left_to_right_distance[left_id][right_id] = torch.sum(torch.abs(self.left[left_id] - self.right[right_id])).item()
#     print("Evaluating right to left")
#     for right_id in self.right.keys():
#       right_to_left_distance[right_id] = {}
#       for left_id in self.left.keys():
#         right_to_left_distance[right_id][left_id] = torch.sum(torch.abs(self.right[right_id] - self.left[left_id])).item()
#
#     for left_id in self.left.keys():
#       sorted_list = sorted(left_to_right_distance[left_id].items(), key=lambda x: x[1])
#       candidate_ids = [x[0] for x in sorted_list][:self.topk]
#       if self.left_to_right_gold[left_id] in candidate_ids:
#         self._n_hit_left += 1
#       # if dump_to_file:
#       #   for right_id, dist in sorted_list:
#       #     fout.write("{}\t{}\t{}\n".format(left_id, right_id, dist))
#     for right_id in self.right.keys():
#       sorted_list = sorted(right_to_left_distance[right_id].items(), key=lambda x: x[1])
#       candidate_ids = [x[0] for x in sorted_list][:self.topk]
#       if self.right_to_left_gold[right_id] in candidate_ids:
#         self._n_hit_right += 1
#       # if dump_to_file:
#       #   for left_id, dist in sorted_list:
#       #     fout.write("{}\t{}\t{}\n".format(right_id, left_id, dist))
#
#     if self.dump_to_file_path:
#       self.dump_to_file_handler.close()
#
#     self._n_total_left = len(self.left_to_right_gold)
#     self._n_total_right = len(self.right_to_left_gold)
#     return [("hits_left", self._n_hit_left),
#             ("hits_right", self._n_hit_right),
#             ("total_left", self._n_total_left),
#             ("total_right", self._n_total_right),
#             ("recall_left", self._n_hit_left / self._n_total_left),
#             ("recall_right", self._n_hit_right / self._n_total_right)]

class RetrievalScorer(Scorer):
  """

  """
  def __init__(self, label_mapping, qrels_file_path, correct_label='1', dump_to_file=None, ):
    super(RetrievalScorer, self).__init__()
    self.label_mapping = label_mapping
    self._inv_label_mapping = {v: k for k, v in label_mapping.items()}
    self._examples = []
    self._preds = []
    self.correct_label = correct_label
    self.qrels_file_path = qrels_file_path
    # Because we need to leverage trec_eval to calculate the scores, so dump_to_file can not be None
    if dump_to_file:
      self.dump_to_file_path = os.path.join(dump_to_file["output_dir"], dump_to_file["task_name"] + "_dump.json")
      self.dump_to_file_handler = open(self.dump_to_file_path, 'w')
    else:
      print("You need to specify the dump_to_file path for the retrieval task")
      exit()

  def update(self, mbs, predictions, loss, extra):
    super(RetrievalScorer, self).update(mbs, predictions, loss, extra)
    # TODO: we are going to migrate the interface !
    predictions = predictions["logits"]
    for example, preds in zip(mbs.examples, predictions):
      self._examples.append(example)
      self._preds.append(preds.data.cpu().numpy())

  def get_loss(self):
    return 0

  def _get_results(self):
    topic_doc_collection = {}
    for example, preds in zip(self._examples, self._preds):
      preds = softmax(preds)
      score = preds[self.label_mapping[self.correct_label]]
      text_a_id, text_b_id = example.guid.split('-')
      if text_a_id not in topic_doc_collection:
        topic_doc_collection[text_a_id] = {}
      topic_doc_collection[text_a_id][text_b_id] = max(topic_doc_collection[text_a_id].get(text_b_id, 0), score)
    for text_a_id in topic_doc_collection:
      for text_b_id in topic_doc_collection[text_a_id]:
        score = topic_doc_collection[text_a_id][text_b_id]
        self.dump_to_file_handler.write("{} Q0 {} 0 {} rerank\n".format(text_a_id, text_b_id, score))

    self.dump_to_file_handler.flush()
    self.dump_to_file_handler.close()
    dir = os.path.abspath(os.path.dirname(__file__))
    trec_eval_path = os.path.join(dir, '..', '..', '..', 'evals', 'trec_eval', 'trec_eval.9.0.4/trec_eval')

    trec_out = subprocess.check_output([trec_eval_path, self.qrels_file_path , self.dump_to_file_path])
    trec_out_lines = str(trec_out, 'utf-8').split('\n')
    mean_average_precision = float(trec_out_lines[5].split('\t')[-1])
    # mean_reciprocal_rank = float(trec_out_lines[9].split('\t')[-1])
    # p_30 = float(trec_out_lines[25].split('\t')[-1])

    return [("map", mean_average_precision)]



