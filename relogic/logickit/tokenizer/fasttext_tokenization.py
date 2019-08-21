import logging
import collections
from relogic.logickit.tokenizer.tokenization import BasicTokenizer, load_vocab

from relogic.utils.file_utils import cached_path

logger = logging.getLogger(__name__)

PRETRAINED_VECTOR_ARCHIVE_MAP = {
  'wiki-news-300d-1M': "https://github.com/Impavidity/relogic/raw/master/relogic/logickit/vocabs/wiki-news-300d-1M.txt",
}

class FasttextTokenizer(object):
  """

  """

  def __init__(self, vocab_file, do_lower_case=True):
    self.vocab = load_vocab(vocab_file)

    self.ids_to_tokens = collections.OrderedDict(
      [(ids, tok) for tok, ids in self.vocab.items()])
    self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)


  def tokenize(self, text):
    tokens, _ = self.basic_tokenizer.tokenize(text)
    return tokens

  def convert_tokens_to_ids(self, tokens):
    """Converts a sequence of tokens into ids using the vocab."""
    ids = []
    for token in tokens:
      ids.append(self.vocab.get(token, 0))
    return ids

  def convert_ids_to_tokens(self, ids):
    """Converts a sequence of ids in fasttext tokens using the vocab."""
    tokens = []
    for i in ids:
      tokens.append(self.ids_to_tokens[i])
    return tokens


  @classmethod
  def from_pretrained(cls, pretrained_name_or_path, cache_dir=None, *inputs, **kwargs):
    if pretrained_name_or_path in PRETRAINED_VECTOR_ARCHIVE_MAP:
      vocab_file = PRETRAINED_VECTOR_ARCHIVE_MAP[pretrained_name_or_path]
    else:
      vocab_file = pretrained_name_or_path
    try:
      resolved_vocab_file = cached_path(vocab_file, cache_dir=cache_dir)
    except EnvironmentError:
      logger.error(
        "Model name '{}' was not found in model name list ({}). "
        "We assumed '{}' was a path or url but couldn't find any file "
        "associated to this path or url.".format(
          pretrained_name_or_path,
          ', '.join(PRETRAINED_VECTOR_ARCHIVE_MAP.keys()),
          vocab_file))
      return None
    if resolved_vocab_file == vocab_file:
      logger.info("loading vocabulary file {}".format(vocab_file))
    else:
      logger.info("loading vocabulary file {} from cache at {}".format(
        vocab_file, resolved_vocab_file))
    tokenizer = cls(resolved_vocab_file, *inputs, **kwargs)
    return tokenizer
