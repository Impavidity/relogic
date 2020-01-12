from relogic.components.component import Component
from relogic.structures.structure import Structure
from typing import List
import spacy

nlp = spacy.load("en_core_web_sm")

class PredicateDetectionComponent(Component):
  """

  """
  def __init__(self, model_name, predictor):
    super().__init__(None, predictor)
    self.model_name = model_name

  @classmethod
  def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
    if pretrained_model_name_or_path == "spacy":
      return PredicateDetectionComponent(model_name=pretrained_model_name_or_path,
                                         predictor=spacy.load("en_core_web_sm", disable=["parser", "ner"]))

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
    for structure in inputs:
      results = self.remerge_sent(self._predictor(structure.text))
      if self.model_name == "spacy":
        for idx, token in enumerate(results):
          structure.add_token(token.text)
          structure.pos.append(token.pos_)
          if token.pos_ == "VERB":
            structure.predicates.append((idx, token.text))