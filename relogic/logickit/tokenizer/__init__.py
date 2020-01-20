from .fasttext_tokenization import FasttextTokenizer
from .tokenizer_roberta_xlm import RobertaXLMTokenizer
from .tokenization import BertTokenizer as CustomizedBertTokenizer
from transformers.tokenization_bert import BertTokenizer

NAME_TO_TOKENIZER_MAP = {
  "xlmr": RobertaXLMTokenizer,
  "fasttext": FasttextTokenizer,
  "customized_bert": CustomizedBertTokenizer,
  "bert": BertTokenizer
}