from dataclasses import dataclass, field
from typing import List, Union

from relogic.structures.structure import Structure

@dataclass
class Pointwise(Structure):
  text_a: str = None
  text_b: str = None