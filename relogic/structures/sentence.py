from dataclasses import dataclass, field
import regex as re
from typing import List, Union

from relogic.structures.structure import Structure
from relogic.structures.token import Token
from relogic.structures.span import Span
from transformers.tokenization_bert import BasicTokenizer

basic_tokenizer = BasicTokenizer(do_lower_case=False)

PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

@dataclass
class Sentence(Structure):
  idx : int = None
  text: str = None
  tokens: List[Token] = field(default_factory=list)
  text_: str = None
  pos: List = field(default_factory=list)
  spans: List[Span] = field(default_factory=list)
  predicate_text: str = None
  predicate_index: int = None
  predicates: List = field(default_factory=list)
  srl_labels: List = field(default_factory=list)

  tokenizer: str = "space"

  def __post_init__(self):
    if self.text:
      if self.tokenizer == "space":
        self.tokens = [Token(token) for token in self.text.split(" ")]
      elif self.tokenizer == "basic":
        self.tokens = [Token(token) for token in basic_tokenizer.tokenize(self.text)]
      elif self.tokenizer == "gpt2":
        self.tokens = [Token(token.strip(" ")) for token in re.findall(PAT, self.text)]
      else:
        raise ValueError("Unknown tokenizer method {}".format(self.tokenizer))
    # if self.text:
    #   # Currently we only use split by tokens
    #   self.tokens = [Token(token) for token in self.text.split()]

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

  @property
  def tokenized_text(self):
    if self.text_ is None:
      # if len(self.tokens) == 0:
        # raise ValueError("The sentence {} is not tokenized.".format(self.text))

      self.text_ = " ".join([token.text for token in self.tokens])
    return self.text_

  def convert_to_json(self):
    return {
      "text": self.tokenized_text,
      "predicates": self.predicates,
      "srl_labels": self.srl_labels
    }

  def __getitem__(self, item):
    return self.tokens[item]
