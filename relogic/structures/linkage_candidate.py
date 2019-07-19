from dataclasses import dataclass
from typing import Dict, List

from relogic.structures.structure import Structure


@dataclass
class LinkageCandidate(Structure):
  """
  """
  text: str
  score: float
  prior: Dict
  alias_of: List
  
  @classmethod
  def from_hit(cls, hit):
    prior_table = {}
    if hit.prior:
      priors = hit.prior.strip().split("\t")
      for uri, count in zip(priors[::2], priors[1::2]):
        uri = uri.replace("WikiProject_", "")
        # TODO: We will unified the cleaning process and add it to utils
        prior_table[uri] = prior_table.get(uri, 0) + int(count)
    alias_of = hit.alias.split("\t")
    return cls(text=hit.content, score=hit.score, prior=prior_table, alias_of=alias_of)
