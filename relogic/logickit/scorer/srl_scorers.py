from relogic.logickit.scorer.tagging_scorers import F1Score, softmax
from relogic.logickit.utils.utils import get_span_labels
from relogic.logickit.scorer.span_level_scorer import SpanLevelScorer
from relogic.logickit.scorer.scorer import Scorer
import os
import torch
import json
from relogic.logickit.dataflow.srl import SRLExample

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
      example: SRLExample
      preds_tags = preds.argmax(-1).data.cpu().numpy()
      confidences = [max(softmax(token_level)) for token_level in preds.data.cpu().numpy()]
      sent_spans, sent_labels = get_span_labels(
        sentence_tags = example.seq_labels,
        )
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
        if len(example.raw_tokens) != len(sent_labels) or len(example.raw_tokens) != len(pred_labels):
          print(len(example.raw_tokens), example.raw_tokens)
          print(len(sent_labels), sent_labels)
          print(len(pred_labels), pred_labels)
          exit()
        self.dump_to_file_handler.write(json.dumps({
          "text": example.raw_tokens,
          "predicate_text": example.predefined_predicate,
          "predicate_index": example.predicate_index,
          "label": sent_labels,
          "predicted": pred_labels
        }) + "\n")
        # data_to_dump.append(((int(example.guid), int(example.predicate_index)), example.raw_text, sent_labels, pred_labels, confidences))

    if self.dump_to_file_path:
      self.dump_to_file_handler.close()

    return super(SRLF1Scorer, self)._get_results()

class JointSpanSRLF1Scorer(Scorer):
  def __init__(self, label_mapping, dump_to_file=None):
    super(JointSpanSRLF1Scorer, self).__init__()
    self._inv_label_mapping = {v: k for k, v in label_mapping.items()}
    self._examples = []
    self._predictions = []
    self._n_correct = 0
    self._n_gold = 0
    self._n_predicted = 0
    if dump_to_file:
      self.dump_to_file_path = os.path.join(dump_to_file["output_dir"], dump_to_file["task_name"] + "_dump.json")
      self.dump_to_file_handler = open(self.dump_to_file_path, 'w')

  def update(self, mbs, predictions, loss, extra_args):
    # predictions = srl_scores, top_pred_spans, top_arg_spans
    srl_scores, top_pred_spans, top_arg_spans, top_pred_span_mask, top_arg_span_mask, _, _, _ = predictions
    # top_pred_spans = (batch, max_seq_len, 2)
    batch_size = top_pred_spans.size(0)

    max_pred_num = top_pred_spans.size(1)
    max_arg_num = top_arg_spans.size(1)
    expanded_top_pred_spans = top_pred_spans.unsqueeze(2).repeat(1, 1, max_arg_num, 1)
    expanded_top_arg_spans = top_arg_spans.unsqueeze(1).repeat(1, max_pred_num, 1, 1)
    expanded_top_pred_span_mask = top_pred_span_mask.unsqueeze(2).repeat(1, 1, max_arg_num)
    expanded_top_arg_span_mask = top_arg_span_mask.unsqueeze(1).repeat(1, max_pred_num, 1)
    indices_mask = expanded_top_pred_span_mask & expanded_top_arg_span_mask

    indices = torch.cat([expanded_top_pred_spans, expanded_top_arg_spans], dim=-1)
    # batch_id = torch.arange(0, batch_size).unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, *indices.size()[1:3], 1).to(
    #   indices.device)
    # expanded_indices = torch.cat([batch_id, indices], dim=-1)
    # flatten_expanded_indices = expanded_indices.view(-1, 5)

    preds_tags = srl_scores.argmax(-1)
    if len(expanded_top_pred_spans) != len(expanded_top_arg_spans) or len(preds_tags) != len(expanded_top_arg_spans):
      raise ValueError("The length of predicate span {} is not equal to argument span {}".format(
        len(expanded_top_pred_spans), len(expanded_top_arg_spans)))

    for example_indices, example_pred, example_indices_mask in zip(indices, preds_tags, indices_mask):
      prediction_list = []
      for indices, pred, mask in zip(example_indices.view(-1, 4), example_pred.view(-1), example_indices_mask.view(-1)):
        if mask:
          pred_label = self._inv_label_mapping[pred.data.cpu().numpy().item()]
          # if pred_label != 'X' and pred_label != 'O':
          prediction_list.append(tuple(indices.data.cpu().numpy()) +  (pred_label,))
      self._predictions.append(prediction_list)
    for example in mbs.examples:
      self._examples.append(example)



  def _get_results(self):
    assert len(self._examples) == len(self._predictions)
    for example, preds in zip(self._examples, self._predictions):
      # sent_spans = set([(pred_start, pred_end, arg_start, arg_end, self._inv_label_mapping[label_id]) for
      #               pred_start, pred_end, arg_start, arg_end, label_id in example.label_ids])
      # TODO: fix for LSTM
      sent_spans = set([(pred_start, pred_end, arg_start, arg_end, self._inv_label_mapping[label_id]) for
                        pred_start, pred_end, arg_start, arg_end, label_id in example.label_ids])
      example: SRLExample
      without_filter_span_preds = set(preds)
      sent_spans = set(filter(lambda item: item[4] != 'V', sent_spans))
      span_preds = set(filter(lambda item: item[4] not in ['V', "O", "X"], without_filter_span_preds))
      self._n_correct += len(sent_spans & span_preds)
      self._n_gold += len(sent_spans)
      self._n_predicted += len(span_preds)

      if self.dump_to_file_path:
        sent_spans = [(int(s[0]), int(s[1]), int(s[2]), int(s[3]), s[4]) for s in sent_spans]
        span_preds = [(int(s[0]), int(s[1]), int(s[2]), int(s[3]), s[4]) for s in span_preds]
        without_filter_span_preds = [(int(s[0]), int(s[1]), int(s[2]), int(s[3]), s[4]) for s in without_filter_span_preds]
        self.dump_to_file_handler.write(json.dumps({
          "text": example.raw_tokens,
          "BPE_tokens": example.text_tokens,
          "span_preds": list(span_preds),
          "sent_spans": list(sent_spans),
          "pruned_span_preds": list(without_filter_span_preds)}) + "\n")
    if self.dump_to_file_path:
      self.dump_to_file_handler.close()

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