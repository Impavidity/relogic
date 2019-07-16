from dataclasses import dataclass
from typing import List

from relogic.structures.structure import Structure
from relogic.structures.sentence import Sentence


@dataclass
class Document(Structure):
  sentences: List[Sentence]