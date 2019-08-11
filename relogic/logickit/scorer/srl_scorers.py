from relogic.logickit.scorer.tagging_scorers import F1Score, softmax
from relogic.logickit.utils.utils import get_span_labels
from relogic.logickit.scorer.span_level_scorer import SpanLevelScorer
import os
import json

class SpanSRLF1Scorer(SpanLevelScorer):
  def __init__(self, label_mapping, dump_to_file=None):
    super(SpanSRLF1Scorer, self).__init__()
    self._inv_label_mapping = {v:k for k, v in label_mapping.items()}
    if dump_to_file:
      self.dump_to_file_path = os.path.join(dump_to_file["output_dir"], dump_to_file["task_name"] + "_dump.json")
      self.dump_to_file_handler = open(self.dump_to_file_path, 'w')

  def _get_results(self):
    self._n_correct, self._n_predicted, self._n_gold = 0, 0, 0
    for example, boundary, pred_label in zip(self._examples, self._boundaries, self._preds):
      pred_tags = pred_label.argmax(-1).data.cpu().numpy()

      sent_spans, sent_labels = get_span_labels(
        sentence_tags=example.label)

      # sent_spans = set([(b[0], b[1], l) for b, l in zip(example.spans, example.span_labels)])
      span_preds = set([(int(b[0]), int(b[1]), self._inv_label_mapping[l]) for b, l in zip(example.span_candidates, pred_tags) if b[0] >= 0])
      span_preds = set(filter(lambda item: item[0] != example.predicate_index, span_preds))
      self._n_correct += len(sent_spans & span_preds)
      self._n_gold += len(sent_spans)
      self._n_predicted += len(span_preds)
      if self.dump_to_file_path:
        self.dump_to_file_handler.write(json.dumps({
          "text": example.raw_text,
          "predicate_text": example.predicate_text,
          "predicate_index": example.predicate_index,
          "span_preds": list(span_preds),
          "sent_spans": list(sent_spans)}) + "\n")
    if self.dump_to_file_path:
      self.dump_to_file_handler.close()
    # TODO: a quick fix for calculating the F1 score
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
  def get_loss(self):
    return 0

class SRLF1Scorer(F1Score):
  def __init__(self, label_mapping, dump_to_file=None):
    super(SRLF1Scorer, self).__init__()
    # label mapping : str -> int
    self._inv_label_mapping = {v: k for k, v in label_mapping.items()}
    if dump_to_file:
      self.dump_to_file_path = os.path.join(dump_to_file["output_dir"], dump_to_file["task_name"] + "_dump.json")
      self.dump_to_file_handler = open(self.dump_to_file_path, 'w')

  def _get_results(self):
    # The scorer will be cleared after the evaluation is done. 

    self._n_correct, self._n_predicted, self._n_gold = 0, 0, 0
    for example, preds in zip(self._examples, self._preds):
      preds_tags = preds.argmax(-1).data.cpu().numpy()
      confidences = [max(softmax(token_level)) for token_level in preds.data.cpu().numpy()]
      sent_spans, sent_labels = get_span_labels(
        sentence_tags = example.label)
      sent_spans = set(filter(lambda item: item[0] != example.predicate_index, sent_spans))
      span_preds, pred_labels = get_span_labels(
        sentence_tags=preds_tags,
        is_head = example.is_head,
        segment_id = example.segment_ids,
        inv_label_mapping = self._inv_label_mapping)
      span_preds = set(filter(lambda item: item[0] != example.predicate_index, span_preds))
      self._n_correct += len(sent_spans & span_preds)
      self._n_gold += len(sent_spans)
      self._n_predicted += len(span_preds)
      if self.dump_to_file_path:
        if len(example.raw_text) != len(sent_labels) or len(example.raw_text) != len(pred_labels):
          print(len(example.raw_text), example.raw_text)
          print(len(sent_labels), sent_labels)
          print(len(pred_labels), pred_labels)
          exit()
        self.dump_to_file_handler.write(json.dumps({
          "text": example.raw_text,
          "predicate_text": example.predicate_text,
          "predicate_index": example.predicate_index,
          "label": sent_labels,
          "predicted": pred_labels
        }) + "\n")
        # data_to_dump.append(((int(example.guid), int(example.predicate_index)), example.raw_text, sent_labels, pred_labels, confidences))

    if self.dump_to_file_path:
      self.dump_to_file_handler.close()

    return super(SRLF1Scorer, self)._get_results()