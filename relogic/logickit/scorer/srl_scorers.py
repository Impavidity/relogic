from relogic.logickit.scorer.tagging_scorers import F1Score, softmax
from relogic.logickit.utils.utils import get_span_labels
import os
import json


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
    data_to_dump = []
    for example, preds in zip(self._examples, self._preds):
      preds_tags = preds.argmax(-1).data.cpu().numpy()
      confidences = [max(softmax(token_level)) for token_level in preds.data.cpu().numpy()]
      sent_spans, sent_labels = get_span_labels(
        sentence_tags = example.label)
      span_preds, pred_labels = get_span_labels(
        sentence_tags=preds_tags,
        is_head = example.is_head,
        segment_id = example.segment_ids,
        inv_label_mapping = self._inv_label_mapping)
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
          "predicted": pred_labels,
          "confidence": confidences
        }) + "\n")
        # data_to_dump.append(((int(example.guid), int(example.predicate_index)), example.raw_text, sent_labels, pred_labels, confidences))

    if self.dump_to_file_path:
      # data_to_dump = sorted(data_to_dump, key=lambda x: x[0])
      # for idx, raw_text, sent_labels, pred_labels, confidences in data_to_dump:
      #   for word, gold, pred, confidence in zip(raw_text, sent_labels, pred_labels, confidences):
      #     self.dump_to_file_handler.write("{} {} {} {}\n".format(word, gold, pred, confidence))
      #   self.dump_to_file_handler.write("\n")
      self.dump_to_file_handler.close()

    return super(SRLF1Scorer, self)._get_results()