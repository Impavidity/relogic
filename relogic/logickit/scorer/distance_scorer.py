import abc
from relogic.logickit.scorer.scorer import Scorer


class DistanceScorer(Scorer):
  def __init__(self):
    super(DistanceScorer, self).__init__()
    self.loss = 0
    self.batch_count = 0

  def update(self, mbs, predictions, loss, extra_args):
    super(DistanceScorer, self).update(mbs, predictions, loss, extra_args)
    self.loss += loss
    self.batch_count += 1

  def get_loss(self):
    return self.loss / self.batch_count

  def _get_results(self):
    return [("distance", - self.loss / self.batch_count)]
