from relogic.components.component import Component
from relogic.structures.structure import Structure
from typing import List


class PredicateDetectionComponent(Component):
  """

  """


  @classmethod
  def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
    import spacy

    if pretrained_model_name_or_path == "spacy":
      return PredicateDetectionComponent(model_name=pretrained_model_name_or_path, config=None,
                                         predictor=spacy.load("en_core_web_sm", disable=["parser", "ner"]))
    else:
      return super().from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)

  def remerge_sent(self, sent):
    i = 0
    while i < len(sent) - 1:
      tok = sent[i]
      if not tok.whitespace_:
        ntok = sent[i + 1]
        # in-place operation.
        sent.merge(tok.idx, ntok.idx + len(ntok))
      i += 1
    return sent

  def execute(self, inputs: List[Structure]):
    if self.model_name == "spacy":
      for structure in inputs:
        results = self.remerge_sent(self._predictor(structure.tokenized_text))
        for idx, token in enumerate(results):
          structure.pos.append(token.pos_)
          if token.pos_ == "VERB":
            structure.predicates.append((idx, token.text))
    else:
      counter = 0
      for results in self._predictor.predict(inputs):
        _, batch_labels, _ = results
        for labels in batch_labels:
          for idx, label in enumerate(labels):
            if label == "B-V":
              inputs[counter].predicates. append((idx, inputs[counter][idx].text))
          counter += 1
