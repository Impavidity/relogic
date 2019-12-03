import abc

from relogic.logickit.scorer.word_level_scorer import WordLevelScorer
from relogic.logickit.utils.utils import get_span_labels, softmax, filter_head_prediction
import os
import json



class AccuracyScorer(WordLevelScorer):
  def __init__(self, label_mapping, ignore_list=None, dump_to_file=None):
    super(AccuracyScorer, self).__init__()
    self.label_mapping = label_mapping
    self._inv_label_mapping = {v: k for k, v in label_mapping.items()}
    self.ignore_list = ignore_list if ignore_list else []
    if dump_to_file:
      self.dump_to_file_path = os.path.join(dump_to_file["output_dir"], dump_to_file["task_name"] + "_dump.json")

  def _get_results(self):
    if self.dump_to_file_path:
      self.dump_to_file_handler = open(self.dump_to_file_path, 'w')

    correct, count = 0, 0
    for example, preds in zip(self._examples, self._preds):
      confidences = [max(softmax(token_level)) for token_level in preds.data.cpu().numpy()]
      preds = preds.argmax(-1).data.cpu().numpy()
      preds = [self._inv_label_mapping[y_pred] for y_pred in preds]
      preds = filter_head_prediction(sentence_tags=preds, is_head=example.is_head)
      assert len(example.labels) == len(preds)
      for y_true, y_pred in zip(example.labels, preds):
        if y_true not in self.ignore_list:
          count += 1
          correct += 1 if y_pred == y_true else 0

      if self.dump_to_file_path:
        self.dump_to_file_handler.write(
          json.dumps({
            "tokens": example.raw_tokens,
            "labels": example.labels,
            "predicted_labels": preds}) + "\n")

    if self.dump_to_file_path:
      self.dump_to_file_handler.close()

    return [
      ("accuracy", 100.0 * correct / count),
      ("loss", self.get_loss())
    ]

class F1Score(WordLevelScorer, metaclass=abc.ABCMeta):
  def __init__(self):
    super(F1Score, self).__init__()
    self._n_correct, self._n_predicted, self._n_gold = 0, 0, 0

  def _get_results(self):
    if self._n_correct == 0:
      p, r, f1 = 0, 0, 0
    else:
      p = 100.0 * self._n_correct / self._n_predicted
      r = 100.0 * self._n_correct / self._n_gold
      f1 = 2 * p * r / (p + r)
    return [
      ("total entity", self._n_gold),
      ("find entity", self._n_predicted),
      ("correct entity", self._n_correct),
      ("precision", p),
      ("recall", r),
      ("f1", f1),
      ("loss", self.get_loss()),
    ]


class EntityLevelF1Scorer(F1Score):
  def __init__(self, label_mapping, dump_to_file=None):
    super(EntityLevelF1Scorer, self).__init__()
    # label mapping : str -> int
    self._inv_label_mapping = {v: k for k, v in label_mapping.items()}
    if dump_to_file:
      self.dump_to_file_path = os.path.join(dump_to_file["output_dir"], dump_to_file["task_name"] + "_dump.json")

  def _get_results(self):
    if self.dump_to_file_path:
      self.dump_to_file_handler = open(self.dump_to_file_path, "w")

    self._n_correct, self._n_predicted, self._n_gold = 0, 0, 0
    for example, preds in zip(self._examples, self._preds):
      preds_tags = preds.argmax(-1).data.cpu().numpy()
      confidences = [max(softmax(token_level)) for token_level in preds.data.cpu().numpy()]
      sent_spans, sent_labels = get_span_labels(
        sentence_tags = example.labels)
      span_preds, pred_labels = get_span_labels(
        sentence_tags=preds_tags,
        is_head = example.is_head,
        segment_id = example.segment_ids,
        inv_label_mapping = self._inv_label_mapping)
      self._n_correct += len(sent_spans & span_preds)
      self._n_gold += len(sent_spans)
      self._n_predicted += len(span_preds)
      if self.dump_to_file_path:
        if len(example.raw_tokens) != len(sent_labels) or len(example.raw_tokens) != len(pred_labels):
          print(len(example.raw_tokens), example.raw_tokens)
          print(len(sent_labels), sent_labels)
          print(len(pred_labels), pred_labels)
          exit()
        self.dump_to_file_handler.write(
          json.dumps({
            "tokens": example.raw_tokens,
            "labels": sent_labels,
            "predicted_labels": pred_labels}) + "\n")

    if self.dump_to_file_path:
      self.dump_to_file_handler.close()

    return super(EntityLevelF1Scorer, self)._get_results()


