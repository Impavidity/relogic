from dataclasses import dataclass, field
from typing import List, Dict, Union

from relogic.structures.structure import Structure
from relogic.structures.token import Token
from relogic.structures.linkage_candidate import LinkageCandidate

@dataclass
class Span(Structure):
  text: str = None
  _text: str = field(init=False, repr=False)
  tokens: List[Token] = field(default_factory=list)
  start_index: int = None
  end_index: int = None

  
  # Linking
  linkage_candidates: List[LinkageCandidate] = field(default_factory=list)
  aggregated_prior: Dict = field(default_factory=dict)
  first_layer_prior: Dict = field(default_factory=dict)
  ranked_uris: List = field(default_factory=list)
  wikipedia_uri: str = None

  def add_linkage_candidate(self, linkage: LinkageCandidate):
    """
    """
    self.linkage_candidates.append(linkage)
    for uri, count in linkage.prior.items():
      self.aggregated_prior[uri] = self.aggregated_prior.get(uri, 0) + count
  
  def add_token(self, token: Union[str, Token]):
    """
    """
    if isinstance(token, str):
      self.tokens.append(Token(text=token))
    else:
      self.tokens.append(token)
    if self._text is not None:
      # Because the sentence has been mutated, the text string
      # needs to be regenerated.
      self._text = None

  def set_start_index(self, start_index: int):
    self.start_index = start_index

  def set_end_index(self, end_index: int):
    self.end_index = end_index

  def add_wikipedia_uri(self, uri: str):
    self.wikipedia_uri = uri

  @property
  def text(self):
    if self._text is None:
      self._text = " ".join([token.text for token in self.tokens])
    return self._text
  
  @text.setter
  def text(self, text):
    if isinstance(text, property):
      # This is a ugly fix for non-arguments initialization
      # s = Span()
      # TODO: find better solution for this implementation
      self._text = None
    else:
      self._text = text