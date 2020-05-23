from dataclasses import dataclass, field
from typing import List

from relogic.structures.structure import Structure

@dataclass
class SQL(Structure):
  idx : int = None
  text : str = None
  columns : List = field(default_factory=list)
