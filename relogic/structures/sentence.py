from dataclasses import dataclass
from typing import List

from relogic.structures.structure import Structure
from relogic.structures.token import Token


@dataclass
class Sentence(Structure):
  tokens: List[Token]