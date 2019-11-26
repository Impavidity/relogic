import torch
import relogic.utils.crash_on_ipy

def clean(text):
  return text.strip()

class RobertaXLMTokenizer:
  def __init__(self, model):
    self.roberta = torch.hub.load('pytorch/fairseq', model)

  @classmethod
  def from_pretrained(cls, pretrained_model_name_or_path):
    return cls(model=pretrained_model_name_or_path)

  def alignment(self, bpe_tokens, expected_tokenization):
    bpe_tokens = [clean(text) for text in bpe_tokens]
    alignment = []
    bpe_toks = filter(lambda item: item[1] != '', enumerate(bpe_tokens, start=1))
    j, bpe_tok = next(bpe_toks)
    for other_tok in expected_tokenization:
      bpe_indices = []
      while True:
        if other_tok.startswith(bpe_tok):
          bpe_indices.append(j)
          other_tok = other_tok[len(bpe_tok):]
          try:
            j, bpe_tok = next(bpe_toks)
          except StopIteration:
            j, bpe_tok = None, None
        elif bpe_tok.startswith(other_tok):
          # other_tok spans multiple BPE tokens
          bpe_indices.append(j)
          bpe_tok = bpe_tok[len(other_tok):]
          other_tok = ''
        else:
          print(bpe_tokens)
          print(bpe_toks)
          print(expected_tokenization)
          raise Exception('Cannot align "{}" and "{}"'.format(other_tok, bpe_tok))
        if other_tok == '':
          break
      assert len(bpe_indices) > 0
      alignment.append(bpe_indices)
    assert len(alignment) == len(expected_tokenization)
    return alignment

  def tokenize_and_add_placeholder_and_convert_to_ids(self, sent, expected_tokenization):
    roberta_ids = self.roberta.encode(sent)
    bpe_tokens = [self.roberta.bpe.decode(self.roberta.task.source_dictionary.string([x])) for x in roberta_ids]

    if expected_tokenization is not None:
      alignmet = self.alignment(bpe_tokens, expected_tokenization)
      is_head = [2]
      for word in alignmet:
        is_head.append(1)
        for _ in word[1:]:
          is_head.append(0)
      is_head.append(2)

    # Fix for empty
    fixed_bpe_tokens = []
    fixed_roberta_ids = []

    for idx, (token, roberta_id) in enumerate(zip(bpe_tokens, roberta_ids.tolist())):

      if token.strip() != "" or idx == 0 or idx == len(bpe_tokens)-1:
        fixed_bpe_tokens.append(token)
        fixed_roberta_ids.append(roberta_id)

    if expected_tokenization is not None:
      assert(len(fixed_bpe_tokens) == len(is_head))
    else:
      is_head = [0] * len(fixed_bpe_tokens)

    return fixed_bpe_tokens, is_head, fixed_roberta_ids

if __name__ == "__main__":
  sent = "286 Hidemichi Tanaka ( Japan ) 66 75 75 70 , Steve Jones 70 69"
  tokenizer = RobertaXLMTokenizer("xlmr.large.v0")
  raise NotImplementedError("Test")