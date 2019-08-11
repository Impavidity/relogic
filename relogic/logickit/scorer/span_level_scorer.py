from relogic.logickit.scorer.scorer import Scorer
import abc


class SpanLevelScorer(Scorer, metaclass=abc.ABCMeta):
  def __init__(self):
    super(SpanLevelScorer, self).__init__()
    self._examples = []
    self._boundaries = []
    self._preds = []

  def update(self, mbs, predictions, loss, extra):
    super(SpanLevelScorer, self).update(mbs, predictions, loss, extra)
    span_boundary_prediction, span_label_prediction = predictions
    start_index, end_index = span_boundary_prediction
    if len(mbs.examples) != len(start_index):
      print("The number of examples is not same as pred examples {} vs {} vs {} vs {}".format(len(mbs.examples),
        len(start_index), len(end_index), len(span_label_prediction)))
      for example in mbs.examples:
        print(example.__dict__)
      print(span_boundary_prediction, span_label_prediction)
      exit()
    for example, boundary_preds_start, boundary_preds_end, label_preds in zip(mbs.examples, start_index, end_index, span_label_prediction):
      self._examples.append(example)
      self._boundaries.append(zip(boundary_preds_start.data.cpu().numpy(), boundary_preds_end.data.cpu().numpy()))
      self._preds.append(label_preds)