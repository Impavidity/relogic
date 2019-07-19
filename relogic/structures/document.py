from dataclasses import dataclass, field
from typing import List

from relogic.structures.structure import Structure
from relogic.structures.sentence import Sentence


@dataclass
class Document(Structure):
  idx: int = None
  sentences: List[Sentence] = field(default_factory=list)

  def add_sentence(self, sentence: Sentence):
    self.sentences.append(sentence)