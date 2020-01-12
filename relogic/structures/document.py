from dataclasses import dataclass, field
from typing import List

from relogic.structures.structure import Structure
from relogic.structures.paragraph import Paragraph
from relogic.structures.sentence import Sentence


@dataclass
class Document(Structure):
  idx: int = None
  text: str = None
  paragraphs: List[Paragraph] = field(default_factory=list)

  def __post_init__(self):
    for para in self.text.split("\n"):
      self.add_paragraph(Paragraph(text=para))

  def add_paragraph(self, paragraph: Paragraph):
    self.paragraphs.append(paragraph)

