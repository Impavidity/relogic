from dataclasses import dataclass, field
from typing import List, Union

from relogic.structures.structure import Structure
from relogic.structures.token import Token
from relogic.structures.span import Span


@dataclass
class Sentence(Structure):
  text: str = None
  tokens: List[Token] = field(default_factory=list)
  spans: List[Span] = field(default_factory=list)

  def add_token(self, token: Union[str, Token]):
    """Append a token into tokens.
    
    Args:
      token (str, Token): 
    """
    if isinstance(token, str):
      self.tokens.append(Token(text=token))
    else:
      self.tokens.append(token)

  def add_span(self, span: Span):
    self.spans.append(span)

  @property
  def length(self):
    return len(self.tokens)