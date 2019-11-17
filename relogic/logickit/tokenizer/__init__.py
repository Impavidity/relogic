from .fasttext_tokenization import FasttextTokenizer
from .tokenizer_roberta_xlm import RobertaXLMTokenizer
from .tokenization import BertTokenizer as CustomizedBertTokenizer

NAME_TO_TOKENIZER_MAP = {
  "xlmr": RobertaXLMTokenizer,
  "fasttext": FasttextTokenizer,
  "customized_bert": CustomizedBertTokenizer
}