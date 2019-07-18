from relogic.logickit.scorer.scorer import Scorer
from itertools import groupby
import collections
import json
from relogic.logickit.scripts import squad_eval
import os

RawResult = collections.namedtuple("RawResult", ["start_logits", "end_logits"])

class SpanExtractionScorer(Scorer):
  def __init__(self, dataset, gold_answer_file, null_score_diff_threshold, dump_to_file=None):
    super(SpanExtractionScorer, self).__init__()
    self.guid_to_example = {}
    self.dataset_name = dataset
    # dataset_json = json.load(open("data/preprocessed_data/{}_dev.json".format(dataset)))
    dataset_json = json.load(open(gold_answer_file))
    self.dataset = dataset_json['data']
    self.null_score_diff_threshold = null_score_diff_threshold

    if dump_to_file:
      self.dump_to_file_path = os.path.join(dump_to_file["output_dir"], dump_to_file["task_name"] + "_dump.json")
      self.dump_to_null_odds_path = os.path.join(dump_to_file["output_dir"], dump_to_file["task_name"] + "_null_odds.json")
      self.dump_to_span_scores_path = os.path.join(dump_to_file["output_dir"], dump_to_file["task_name"] + "_span_scores.json")


  def update(self, mb, predictions, loss, extra_args):
    # Basically one example can map several features
    # How to map it back ?
    # 1. Build guid to example mapping
    # 2. Map the input_feature to example using guid
    for example in mb.examples:
      self.guid_to_example[example.guid] = example
    pair_group = groupby(zip(mb.input_features, predictions[0], predictions[1]), key=lambda pair: pair[0].guid)
    pair_group = {k: list(v) for k, v in pair_group}
    # {guid: list of pairs}
    for guid in pair_group.keys():
      raw_features = []
      raw_results = []
      for item in pair_group[guid]:
        raw_features.append(item[0])
        raw_results.append(RawResult(start_logits=item[1], end_logits=item[2]))
      self.guid_to_example[guid].write_predictions(
        raw_features=raw_features, 
        raw_results=raw_results,
        n_best_size=20,
        with_negative=mb.task_name in ["squad20"])

  def _get_results(self):
    assert self.dump_to_file_path is not None

    if self.dump_to_file_path:
      self.dump_to_file_handler = open(self.dump_to_file_path, 'w')
      self.dump_to_null_odds_handler = open(self.dump_to_null_odds_path, 'w')
      self.dump_to_span_scores_handler = open(self.dump_to_span_scores_path, 'w')

    prediction_json = {}

    for example in self.guid_to_example.values():
      prediction_json[example.guid] = example.prediction
    json.dump(prediction_json, self.dump_to_file_path)
    if self.dataset_name in ["squad20"]:
      scores_diff_json = {}
      for example in self.guid_to_example.values():
        scores_diff_json[example.guid] = example.scores_diff
      json.dump(scores_diff_json, self.dump_to_null_odds_handler)

      span_scores_json = {}
      for example in self.guid_to_example.values():
        span_scores_json[example.guid] = example.span_score
      json.dump(span_scores_json, self.dump_to_span_scores_handler)

    return squad_eval(self.dataset_name)(self.dataset, prediction_json,
              self.null_score_diff_threshold)

  def get_loss(self):
    return 0