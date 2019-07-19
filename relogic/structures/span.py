from dataclasses import dataclass, field
from typing import List, Dict

from relogic.structures.structure import Structure
from relogic.structures.token import Token
from relogic.structures.linkage_candidate import LinkageCandidate

@dataclass
class Span(Structure):
  text: str
  tokens: List[Token] = field(default_factory=list)

  
  # Linking
  linkage_candidates: List[LinkageCandidate] = field(default_factory=list)
  aggregated_prior: Dict = field(default_factory=dict)
  first_layer_prior: Dict = field(default_factory=dict)
  ranked_uris: List = field(default_factory=list)

  def add_linkage_candidate(self, linkage: LinkageCandidate):
    self.linkage_candidates.append(linkage)
    for uri, count in linkage.prior.items():
      self.aggregated_prior[uri] = self.aggregated_prior.get(uri, 0) + count
  