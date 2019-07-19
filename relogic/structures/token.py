from dataclasses import dataclass

from relogic.structures.structure import Structure


@dataclass
class Token(Structure):
  text: str = None