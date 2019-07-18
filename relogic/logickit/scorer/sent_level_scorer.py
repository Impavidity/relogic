import abc
from relogic.logickit.scorer.scorer import Scorer

class SentLevelScorer(Scorer, metaclass=abc.ABCMeta):
  def __init__(self):
    super(SentLevelScorer, self).__init__()
    self._total_loss = 0
    self._total_sents = 0
    self._examples = []
    self._preds = []
    self._attn_maps = []


  # def update(self, mbs, predictions, loss, extra_args):
  #   super(SentLevelScorer, self).update(mbs, predictions, loss, extra_args)
  #   n_sents = 0
  #   for example, preds in zip(mbs.examples, predictions):
  #     self._examples.append(example)
  #     self._preds.append(preds)
  #     # self._attn_maps.append(attention_map)
  #     # do not dumping heavy results on memory
  #     n_sents += 1
  #   self._total_loss += loss * n_sents
  #   self._total_sents += n_sents

  def get_loss(self):
    return self._total_loss / max(1, self._total_sents)