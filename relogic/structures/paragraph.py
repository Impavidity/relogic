from dataclasses import dataclass, field
from typing import List

from relogic.structures.structure import Structure
from relogic.structures.sentence import Sentence
from nltk.tokenize import sent_tokenize

@dataclass
class Paragraph(Structure):
  idx: int = None
  text: str= None
  sentences: List[Sentence] = field(default_factory=list)

  def __post_init__(self):
    for sent in sent_tokenize(self.text):
      self.add_sentence(Sentence(text=sent))

  def add_sentence(self, sentence: Sentence):
    self.sentences.append(sentence)