from dataclasses import dataclass
from typing import List, Dict

from relogic.structures.structure import Structure
from relogic.structures.token import Token
from relogic.structures.linkage_candidate import LinkageCandidate

@dataclass
class Span(Structure):
  tokens: List[Token]
  text: str
  
  # Linking
  linkage_candidates: List[LinkageCandidate] = []
  aggregated_prior: Dict = {}

  def add_linkage_candidate(self, linkage: LinkageCandidate):
    self.linkage_candidates.append(linkage)
    for uri, count in linkage.prior.items():
      self.aggregated_prior[uri] = self.aggregated_prior.get(uri, 0) + count
  