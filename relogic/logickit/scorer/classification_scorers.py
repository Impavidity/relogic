import abc
from relogic.logickit.scorer.sent_level_scorer import SentLevelScorer
from relogic.logickit.utils.utils import softmax
import json
import os


class F1Score(SentLevelScorer, metaclass=abc.ABCMeta):
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
      ("total label", self._n_gold),
      ("find label", self._n_predicted),
      ("correct label", self._n_correct),
      ("precision", p),
      ("recall", r),
      ("f1", f1),
      ("loss", self.get_loss()),
    ]

class RelationF1Scorer(F1Score):
  def __init__(self, label_mapping, dump_to_file):
    super(RelationF1Scorer, self).__init__()
    self._inv_label_mapping = {v: k for k, v in label_mapping.items()}
    self._o = 'no_relation'
    self.counter = 0
    if dump_to_file:
      self.dump_to_file_path = os.path.join(dump_to_file["output_dir"], dump_to_file["task_name"] + "_dump.json")
      self.dump_to_file_handler = open(self.dump_to_file_path, 'w')


  def update(self, mbs, predictions, loss, extra_args):
    super(RelationF1Scorer, self).update(mbs, predictions, loss, extra_args)
    if self.need_to_clear_output:
      self.dump_to_file_handler = open(self.dump_to_file_path, 'w')
      self.need_to_clear_output = False
    n_sents = 0
    for example, preds in zip(mbs.examples, predictions):
      self._examples.append(example)
      self._preds.append(preds)
      pred_tag = preds.argmax(-1).item()
      if self.dump_to_file_path:
        self.dump_to_file_handler.write(
          json.dumps({
            "text": example.raw_text,
            "subject": example.subj_text,
            "object": example.obj_text,
            "label": example.label,
            "predicted": self._inv_label_mapping[pred_tag]
          }) + "\n")
      # self._attn_maps.append(attention_map)
      # do not dumping heavy results on memory
      n_sents += 1
    self._total_loss += loss * n_sents
    self._total_sents += n_sents

  def _get_results(self):
    self.need_to_clear_output = True

    self._n_correct, self._n_predicted, self._n_gold = 0, 0, 0
    # data_to_dump = []

    for example, preds in zip(self._examples, self._preds):
      pred_tag = preds.argmax(-1).item()

      confidence = max(softmax(preds.data.cpu().numpy()))
      if self._inv_label_mapping[pred_tag] == example.label and \
        example.label != self._o:
          self._n_correct += 1
      if example.label != self._o:
        self._n_gold += 1
      if self._inv_label_mapping[pred_tag] != self._o:
        self._n_predicted += 1
      # if self.dump_to_file_path:
      #   data_to_dump.append((int(example.guid), example.raw_text,
      #                        example.subj_text, example.obj_text, example.label,
      #                        self._inv_label_mapping[pred_tag], confidence))

    # if self.dump_to_file_path:
    #   data_to_dump = sorted(data_to_dump, key=lambda x: x[0])
    #   for idx, raw_text, subj, obj, gold_label, pred_label, confidence in data_to_dump:
    #     json_to_write = {
    #       "text": raw_text,
    #       "subject": subj,
    #       "object": obj,
    #       "relation": gold_label,
    #       "predicted_relation": pred_label,
    #       "confidence": confidence
    #     }
    #     self.dump_to_file_handler.write(json.dumps(json_to_write) + "\n")
    #
    #
    # if self.dump_to_file_path:
    #   self.dump_to_file_handler.close()


    return super(RelationF1Scorer, self)._get_results()