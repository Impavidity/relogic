from relogic.logickit.scorer.scorer import Scorer
import os
import json

class EditNetScorer(Scorer):
  def __init__(self, dump_to_file, dataflow):
    super().__init__()
    self._examples = []
    self._preds = []
    self.dataflow = dataflow
    if dump_to_file:
      self.dump_to_file_path = os.path.join(dump_to_file["output_dir"], dump_to_file["task_name"] + "_dump.json")
      self.dump_to_file_handler = open(self.dump_to_file_path, 'w')
    self._n_correct = 0
    self._n_total = 0

  def update(self, mbs, predictions, loss, extra_args):
    super().update(mbs, predictions, loss, extra_args)
    self._examples.extend(mbs.examples)
    preds = self.dataflow.decode_to_labels(preds=predictions, mb=mbs)
    self._preds.extend(preds)
    if self.dump_to_file_path:
      for example, pred in zip(mbs.examples, preds):
        self.dump_to_file_handler.write(
          json.dumps({
            "idx": example.idx,
            "db_id": example.db_id,
            "text": example.text,
            "candidate": example.sql_candidate,
            "labels": example.label_ids[:-1],
            "predicted": pred,
          }) + "\n")

  def _get_results(self):
    assert len(self._examples) == len(self._preds)
    n_correct = 0

    for example, pred in zip(self._examples, self._preds):
      if pred == example.label_ids[:-1]:
        n_correct += 1
      self._n_total += 1
    return [("accuracy", n_correct / self._n_total * 100.0)]

  def get_loss(self):
    return 0

class SlotFillingScorer(Scorer):
  def __init__(self, dump_to_file, dataflow):
    super().__init__()
    self._examples = []
    self.dataflow = dataflow
    self._column_preds = []
    self._table_preds = []
    if dump_to_file:
      self.dump_to_file_path = os.path.join(dump_to_file["output_dir"], dump_to_file["task_name"] + "_dump.json")
      self.dump_to_file_handler = open(self.dump_to_file_path, 'w')
    self._n_correct = 0
    self._n_total = 0
    self._n_retrieved = 0

  def update(self, mbs, predictions, loss, extra_args):
    super().update(mbs, predictions, loss, extra_args)
    self._examples.extend(mbs.examples)
    column_labels, table_labels = self.dataflow.decode_to_labels(preds=predictions, mb=mbs)
    self._column_preds.extend(column_labels)
    self._table_preds.extend(table_labels)
    if self.dump_to_file_path:
      for example, pred_tables, pred_columns in zip(mbs.examples, table_labels, column_labels):
        self.dump_to_file_handler.write(
          json.dumps({
            "idx": example.idx,
            "text": example.text,
            "query": example.query,
            "db_id": example.db_id,
            "sketch": example.sql,
            "predicted_tables": pred_tables,
            "predicted_columns": pred_columns,
            "gold_tables": example.table_labels,
            "gold_columns": example.column_labels,
            "gold_sql": example.gold_sql
          }) + "\n")

  def _get_results(self):
    assert len(self._examples) == len(self._column_preds)
    assert len(self._examples) == len(self._table_preds)
    table_n_correct = 0
    column_n_correct = 0
    query_n_correct = 0

    for example, pred_columns, pred_tables in zip(self._examples, self._column_preds, self._table_preds):
      c_correct = 0
      t_correct = 0
      if pred_columns == example.column_labels:
        column_n_correct += 1
        c_correct = 1
      table_count = len(example.table_labels)
      for p_tables, g_tables in zip(pred_tables, example.table_labels):
        if set(p_tables) == set(g_tables):
          t_correct += 1
      if t_correct == table_count:
        table_n_correct += 1
      if c_correct == 1 and table_count == t_correct:
        query_n_correct += 1
      self._n_total += 1
    return [("table_accuracy", table_n_correct / self._n_total * 100.0),
            ("column_accuracy", column_n_correct / self._n_total * 100.0),
            ("query_accuracy", query_n_correct / self._n_total * 100.0)]

  def get_loss(self):
    return 0


class RATScorer(Scorer):
  def __init__(self, dump_to_file, dataflow):
    super().__init__()
    self._examples = []
    self._preds_fine_grain = []
    self._preds_sketch = []
    self._preds_deliberate = []
    self.dataflow = dataflow
    if dump_to_file:
      self.dump_to_file_path = os.path.join(dump_to_file["output_dir"], dump_to_file["task_name"] + "_dump.json")
      self.dump_to_file_handler = open(self.dump_to_file_path, 'w')
    self._n_correct = 0
    self._n_total = 0
    self._n_retrieved = 0

  def update(self, mbs, predictions, loss, extra_args):
    super().update(mbs, predictions, loss, extra_args)
    self._examples.extend(mbs.examples)
    (preds_sketch, preds_fine_grain, preds_deliberate,
     preds_sketch_beam, preds_fine_grain_beam, preds_deliberate_beam,
     sketch_beam_scores, fine_grain_beam_scores, deliberate_beam_scores) = self.dataflow.decode_to_labels(
      preds=predictions, mb=mbs)
    self._preds_sketch.extend(preds_sketch)
    self._preds_fine_grain.extend(preds_fine_grain)
    if len(preds_deliberate) > 0:
      self._preds_deliberate.extend(preds_deliberate)
    # self._preds_sketch_beam.extend(preds_sketch_beam)
    # self._preds_fine_grain_beam.extend(preds_fine_grain_beam)
    if self.dump_to_file_path:
      for idx, (example, pred_sketch, pred_fine_grain) in enumerate(zip(mbs.examples, preds_sketch, preds_fine_grain)):
        self.dump_to_file_handler.write(
          json.dumps({
            "idx": example.idx,
            "text": example.text,
            "linked_text": example._linked_input_tokens,
            "query": example.query,
            "labels": example.labels,
            "deliberate_labels": example.deliberate_labels,
            "sketch": example.labels_sketch,
            "predicted_sketch": pred_sketch,
            "predicted": pred_fine_grain,
            "predicted_deliberate": preds_deliberate[idx] if len(preds_deliberate) > 0 else [],
            "predicted_sketch_beam": preds_sketch_beam[idx] if len(preds_sketch_beam) > 0 else [],
            "predicted_beam": preds_fine_grain_beam[idx] if len(preds_fine_grain_beam) > 0 else [],
            "predicted_deliberate_beam": preds_deliberate_beam[idx] if len(preds_deliberate_beam) > 0 else [],
            "sketch_beam_scores": sketch_beam_scores[idx] if len(sketch_beam_scores) > 0 else [],
            "beam_scores": fine_grain_beam_scores[idx] if len(fine_grain_beam_scores) > 0 else [],
            "deliberate_beam_scores": deliberate_beam_scores[idx] if len(deliberate_beam_scores) > 0 else []
          }) + "\n")

  def _get_results(self):
    assert len(self._examples) == len(self._preds_fine_grain)
    assert len(self._examples) == len(self._preds_sketch)
    sketch_n_correct = 0
    fine_grain_n_correct = 0
    deliberate_n_correct = 0

    for example, pred_sketch, pred_fine_grain in zip(self._examples, self._preds_sketch, self._preds_fine_grain):
      if pred_sketch == example.labels_sketch:
        sketch_n_correct += 1
      if pred_fine_grain == example.labels:
        fine_grain_n_correct += 1
      self._n_total += 1
    if len(self._preds_deliberate) > 0:
      assert len(self._examples) == len(self._preds_deliberate)
      for example, pred_deliberate in zip(self._examples, self._preds_deliberate):
        if example.deliberate_labels == pred_deliberate:
          deliberate_n_correct += 1
    return [("sketch_accuracy", sketch_n_correct / self._n_total * 100.0),
            ("fine_grain_accuracy", fine_grain_n_correct / self._n_total * 100.0),
            ("deliberate_accuracy", deliberate_n_correct / self._n_total * 100.0)]

  def get_loss(self):
    return 0



class ColumnSelectionScorer(Scorer):
  def __init__(self, dump_to_file, dataflow):
    super().__init__()
    self._examples = []
    self._preds = []
    self.dataflow = dataflow
    if dump_to_file:
      self.dump_to_file_path = os.path.join(dump_to_file["output_dir"], dump_to_file["task_name"] + "_dump.json")
      self.dump_to_file_handler = open(self.dump_to_file_path, 'w')
    self._n_correct = 0
    self._n_total = 0
    self._n_retrieved = 0

  def update(self, mbs, predictions, loss, extra_args):
    super().update(mbs, predictions, loss, extra_args)
    self._examples.extend(mbs.examples)
    preds = self.dataflow.decode_to_labels(preds=predictions, mb=mbs)
    self._preds.extend(preds)
    if self.dump_to_file_path:
      for example, pred in zip(mbs.examples, preds):
        self.dump_to_file_handler.write(
          json.dumps({
            "text": example.text,
            "candidates": example.candidates,
            "labels": example.labels,
            "predicted": pred[:len(example.labels)]
          }) + "\n")


  def _get_results(self):
    for example, pred in zip(self._examples, self._preds):
      null_label = ["0"] * len(example.labels)
      if example.labels != null_label:
        self._n_total += 1
      if pred[:len(example.labels)] != null_label:
        self._n_retrieved += 1
      if example.labels != null_label and example.labels == pred[:len(example.labels)]:
        self._n_correct += 1
    p = self._n_correct / self._n_retrieved
    r = self._n_correct / self._n_total
    f1 = 2 * p * r / (p + r)
    return [("p", p), ("r", r), ("f1", f1)]


  def get_loss(self):
    return 0

class SQLRerankingScorer(Scorer):
  def __init__(self, dump_to_file, dataflow):
    super().__init__()
    self._examples = []
    self._preds = []
    self.dataflow = dataflow
    if dump_to_file:
      self.dump_to_file_path = os.path.join(dump_to_file["output_dir"], dump_to_file["task_name"] + "_dump.json")
      self.dump_to_file_handler = open(self.dump_to_file_path, 'w')
    self._n_correct = 0
    self._n_total = 0

  def update(self, mbs, predictions, loss, extra_args):
    super().update(mbs, predictions, loss, extra_args)
    self._examples.extend(mbs.examples)
    preds, scores = self.dataflow.decode_to_labels(preds=predictions, mb=mbs)
    self._preds.extend(preds)
    if self.dump_to_file_path:
      for idx, (example, pred) in enumerate(zip(mbs.examples, preds)):
        self.dump_to_file_handler.write(
          json.dumps({
            "idx": example.idx,
            "text": example.text,
            "candidates": example.sql_candidates,
            "labels": example.labels,
            "predicted": pred,
            "scores": scores[idx]
          }) + "\n")
  def _get_results(self):
    for example, pred in zip(self._examples, self._preds):
      self._n_total += 1
      if pred == example.labels:
        self._n_correct += 1
    print("correct: {}, total: {}".format(self._n_correct, self._n_total))
    return [("accuracy", self._n_correct / self._n_total * 100.0)]

  def get_loss(self):
    return 0
