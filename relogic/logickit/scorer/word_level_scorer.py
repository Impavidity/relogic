import abc
from relogic.logickit.scorer.scorer import Scorer

class WordLevelScorer(Scorer, metaclass=abc.ABCMeta):
  def __init__(self):
    super(WordLevelScorer, self).__init__()
    self._total_loss = 0
    self._total_words = 0
    self._examples = []
    self._preds = []

  def update(self, mbs, predictions, loss, extra):
    # TODO: need to double check
    # TODO migrate interface!!!
    predictions = predictions["logits"]
    super(WordLevelScorer, self).update(mbs, predictions, loss, extra)
    n_words = 0
    for example, preds in zip(mbs.examples, predictions):
      self._examples.append(example)
      self._preds.append(preds)
      n_words += len(example.tokens)
    self._total_loss += loss * n_words
    self._total_words += n_words

  def get_loss(self):
    return self._total_loss / max(1, self._total_words)