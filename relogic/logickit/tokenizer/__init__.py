from .fasttext_tokenization import FasttextTokenizer
from .tokenizer_roberta_xlm import RobertaXLMTokenizer as CustomizedRobertaXLMTokenizer
from .tokenization import BertTokenizer as CustomizedBertTokenizer
from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_bart import BartTokenizer
from transformers.tokenization_roberta import RobertaTokenizer
from transformers.tokenization_xlm_roberta import XLMRobertaTokenizer

NAME_TO_TOKENIZER_MAP = {
  "xlmr": CustomizedRobertaXLMTokenizer,
  "fasttext": FasttextTokenizer,
  "customized_bert": CustomizedBertTokenizer,
  "bert": BertTokenizer,
  "bart": BartTokenizer,
  "roberta": RobertaTokenizer,
  "xlm_roberta": XLMRobertaTokenizer,
}