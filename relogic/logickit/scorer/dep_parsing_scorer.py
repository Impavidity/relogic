from relogic.logickit.scorer.scorer import Scorer
import numpy as np
class DepParsingScorer(Scorer):
  def __init__(self, label_mapping, dump_to_file=None):
    super().__init__()
    self.labeled_correct = 0.0
    self.unlabeled_correct = 0.0
    self.exact_labeled_correct = 0.0
    self.exact_unlabeled_correct = 0.0
    self.total_words = 0.0
    self.total_sentences = 0.0

  def _get_results(self):
    """
    Returns
    -------
    The accumulated metrics as a dictionary.
    """
    unlabeled_attachment_score = 0.0
    labeled_attachment_score = 0.0
    unlabeled_exact_match = 0.0
    labeled_exact_match = 0.0
    if self.total_words > 0.0:
      unlabeled_attachment_score = float(self.unlabeled_correct) / float(self.total_words)
      labeled_attachment_score = float(self.labeled_correct) / float(self.total_words)
    if self.total_sentences > 0:
      unlabeled_exact_match = float(self.exact_unlabeled_correct) / float(
        self.total_sentences
      )
      labeled_exact_match = float(self.exact_labeled_correct) / float(self.total_sentences)
    return [
      ("total_sents", self.total_sentences),
      ("total_words", self.total_words),
      ("UAS", unlabeled_attachment_score * 100.0),
      ("LAS", labeled_attachment_score * 100.0),
      ("UEM", unlabeled_exact_match * 100.0),
      ("LEM", labeled_exact_match * 100.0),
    ]


  def update(self, mb, predictions, loss, extra_args):
    predicted_indices = predictions["heads"]
    predicted_labels = predictions["head_tags"]
    # We know there will be mask here
    masks = predictions["mask"]

    predicted_indices, predicted_labels, masks = self.unwrap_to_tensors(predicted_indices, predicted_labels, masks)
    for each_predicted_indices, each_predicted_labels, example, mask in zip(predicted_indices, predicted_labels, mb.examples, masks):
      gold_indices = np.array(example.arcs_ids[1:-1])[np.array(example.is_head[1:-1]) == 1]
      gold_label_ids = np.array(example.label_ids[1:-1])[np.array(example.is_head[1:-1]) == 1]
      each_predicted_indices = each_predicted_indices[1:][mask[1:] == 1].numpy()
      each_predicted_labels = each_predicted_labels[1:][mask[1:] == 1].numpy()

      assert(len(gold_indices) == len(each_predicted_indices))

      correct_indices = (gold_indices == each_predicted_indices)
      correct_labels = (gold_label_ids == each_predicted_labels)
      correct_labels_and_indices = np.logical_and(correct_indices, correct_labels)
      exact_unlabeled_correct = (correct_indices.sum() == len(correct_indices))
      exact_labeled_correct = (correct_labels_and_indices.sum() == len(correct_indices))

      self.unlabeled_correct += correct_indices.sum()
      self.exact_unlabeled_correct += exact_unlabeled_correct
      self.labeled_correct += correct_labels_and_indices.sum()
      self.exact_labeled_correct += exact_labeled_correct
      self.total_sentences += 1
      self.total_words += len(correct_indices)

  def get_loss(self):
    return 0



