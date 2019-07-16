from dataclasses import dataclass
from typing import List

from relogic.structures.document import Document
from relogic.structures.sentence import Sentence


@dataclass
class ReaderDataStructure:
  doc_list: List[Document]
  query: Sentence

