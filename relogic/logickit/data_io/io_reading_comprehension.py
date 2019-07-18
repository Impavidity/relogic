import json
import collections
from relogic.logickit.tokenizer.tokenization import BasicTokenizer
import math

_PrelimPrediction = collections.namedtuple(
  "PrelimPrediction",
  ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

_NbestPrediction = collections.namedtuple(
  "NbestPrediction",
  ["text", "start_logit", "end_logit"])

_Doc_span = collections.namedtuple("DocSpan", ["start", "length"])

# null_score_diff_threshold = -2.2731900215148926
# null_score_diff_threshold = -6.295993328094482
null_score_diff_threshold = 1.0
def _compute_softmax(scores):
  """Compute softmax probability over raw logits."""
  if not scores:
    return []

  max_score = None
  for score in scores:
    if max_score is None or score > max_score:
      max_score = score

  exp_scores = []
  total_sum = 0.0
  for score in scores:
    x = math.exp(score - max_score)
    exp_scores.append(x)
    total_sum += x

  probs = []
  for score in exp_scores:
    probs.append(score / total_sum)
  return probs

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
  answer_tokens, _ = tokenizer.tokenize(orig_answer_text)
  tok_answer_text = ' '.join(answer_tokens)

  for new_start in range(input_start, input_end + 1):
    for new_end in range(input_end, new_start - 1, -1):
      text_span = " ".join(doc_tokens[new_start:(new_end+1)])
      if text_span == tok_answer_text:
        return (new_start, new_end)

  return (input_start, input_end)

def _check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index
  return cur_span_index == best_span_index

def _get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list"""
  index_and_score = sorted(enumerate(logits), key=lambda x:x[1], reverse=True)
  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0]) # (index, score) -> sorted by score -> get the best n index
  return best_indexes

def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
  def _strip_spaces(text):
    ns_chars = []
    ns_to_s_map = collections.OrderedDict()
    for (i, c) in enumerate(text):
      if c == " ":
        continue
      ns_to_s_map[len(ns_chars)] = i
      ns_chars.append(c)
    ns_text = "".join(ns_chars)
    return (ns_text, ns_to_s_map)

  tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
  _tokens, _is_head = tokenizer.tokenize(orig_text)
  tok_text = " ".join(_tokens)

  start_position = tok_text.find(pred_text)
  if start_position == -1:
    if verbose_logging:
      print("Unable to find text: {} in {}".format(pred_text, orig_text))
    return orig_text
  end_position = start_position + len(pred_text) - 1

  (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
  (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

  if len(orig_ns_text) != len(tok_ns_text):
    if verbose_logging:
      print("Length not equal after stripping spaces: {} vs {}".format(orig_ns_text, tok_ns_text))
    return orig_text

  # We then project the characters in `pred_text` back to `orig_text` using
  # the character-to-character alignment.
  tok_s_to_ns_map = {}
  for (i, tok_index) in tok_ns_to_s_map.items():
    tok_s_to_ns_map[tok_index] = i

  orig_start_position = None
  if start_position in tok_s_to_ns_map:
    ns_start_position = tok_s_to_ns_map[start_position]
    if ns_start_position in orig_ns_to_s_map:
      orig_start_position = orig_ns_to_s_map[ns_start_position]

  if orig_start_position is None:
    if verbose_logging:
      print("Couldn't map start position")
    return orig_text

  orig_end_position = None
  if end_position in tok_s_to_ns_map:
    ns_end_position = tok_s_to_ns_map[end_position]
    if ns_end_position in orig_ns_to_s_map:
      orig_end_position = orig_ns_to_s_map[ns_end_position]

  if orig_end_position is None:
    if verbose_logging:
      print("Couldn't map end position")
    return orig_text

  output_text = orig_text[orig_start_position:(orig_end_position + 1)]
  return output_text


class ReadingComprehensionExample(object):
  def __init__(self, guid, text_a, text_b, start_position, end_position, is_impossible, orig_answer_text):
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.start_position = start_position
    self.end_position = end_position
    self.raw_doc = text_b.split()
    self.is_impossible = is_impossible
    self.orig_answer_text = orig_answer_text
    self.is_training = (self.start_position is not None) or (self.end_position is not None)
    self.is_valid = True

    # Because we create several segments for one long document
    # So in one example, we use list to store input_ids, input_mask and segment_ids information
    # This will also affect the process of converting example to features
    self.input_tokens_list = []
    self.input_ids_list = []
    self.input_mask_list = []
    self.segment_ids_list = []
    self.start_position_list = []
    self.end_position_list = []
    self.answer_list = []
    self.token_to_orig_map_list = []
    self.token_is_max_context_list = []

    # We just write the prediction results to example ?
    # Let's try to enable the write function with example.
    # The RawResults object is an argument to that function.

    self.max_span_length = 0
    self.sum_span_length = 0
    self.raw_text_length = len(text_a.split()) + len(text_b.split())

  def __str__(self):
    return str(self.__dict__)

  def process(self, tokenizer, extra_args=None):
    self.text_a_tokens, self.text_a_is_head = tokenizer.tokenize(self.text_a)
    self.text_b_tokens, self.text_b_is_head = tokenizer.tokenize(self.text_b)
    # self.text_b_tokens corresponds to all_doc_tokens in run_squad.py
    assert "max_query_length" in extra_args
    assert "max_seq_length" in extra_args
    max_query_length = extra_args["max_query_length"]
    max_seq_length = extra_args["max_seq_length"]

    if len(self.text_a_tokens) > max_query_length:
      self.text_a_tokens = self.text_a_tokens[:max_query_length]
      self.text_a_is_head = self.text_a_is_head[:max_query_length]

    # The -3 accounts for [CLS], [SEP], and [SEP]
    max_doc_length = max_seq_length - len(self.text_a_tokens) - 3
    # map from the original tokens to index
    orig_to_tok_index = [idx for idx, value in enumerate(self.text_b_is_head) if value == 1]
    tok_to_orig_index = []
    word_index = -1
    for indicator in self.text_b_is_head:
      if indicator == 1:
        word_index += 1
      tok_to_orig_index.append(word_index)

    tok_start_position = None
    tok_end_position = None
    if self.is_training and self.is_impossible:
      tok_start_position = -1
      tok_end_position = -1
    if self.is_training and not self.is_impossible:
      tok_start_position = orig_to_tok_index[self.start_position]
      if self.end_position < len(self.raw_doc) - 1:
        tok_end_position = orig_to_tok_index[self.end_position + 1] - 1
      else:
        tok_end_position = len(self.text_b_tokens) - 1
      (tok_start_position, tok_end_position) = _improve_answer_span(
        self.text_b_tokens, tok_start_position, tok_end_position, tokenizer, self.orig_answer_text)

    if self.is_training and not self.is_impossible:
      new_ans = " ".join(self.text_b_tokens[tok_start_position: tok_end_position+1])
      if not (new_ans[0].lower() == self.orig_answer_text[0].lower() and new_ans[-1].lower() == self.orig_answer_text[-1].lower()):
        self.is_valid = False
        return

    assert "doc_stride" in extra_args
    doc_stride = extra_args["doc_stride"]
    doc_spans = []
    start_offset = 0
    while start_offset < len(self.text_b_tokens):
      length = len(self.text_b_tokens) - start_offset
      if length > max_doc_length:
        length = max_doc_length
      doc_spans.append(_Doc_span(start=start_offset, length=length))
      if start_offset + length == len(self.text_b_tokens):
        break
      start_offset += min(length, doc_stride)
    
    for (doc_span_index, doc_span) in enumerate(doc_spans):
      tokens = []
      token_to_orig_map = {}
      token_is_max_context = {}
      segment_ids = []
      tokens.append("[CLS]")
      segment_ids.append(0)
      for token in self.text_a_tokens:
        tokens.append(token)
        segment_ids.append(0)
      tokens.append("[SEP]")
      segment_ids.append(0)

      for i in range(doc_span.length):
        split_token_index = doc_span.start + i
        token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
        is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
        token_is_max_context[len(tokens)] = is_max_context
        tokens.append(self.text_b_tokens[split_token_index])
        segment_ids.append(1)
      tokens.append("[SEP]")
      segment_ids.append(1)

      input_ids = tokenizer.convert_tokens_to_ids(tokens)

      # Processing the answer
      start_position = None
      end_position = None
      if self.is_training and not self.is_impossible:
        # For training, if our document chunk does not contain an annotation
        # we throw it out, since there is nothing to predict
        doc_start = doc_span.start
        doc_end = doc_span.start + doc_span.length - 1
        out_of_span = False
        if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
          out_of_span = True
        if out_of_span:
          start_position = 0
          end_position = 0
        else:
          doc_offset = len(self.text_a_tokens) + 2 # [CLS] and [SEP]
          start_position = tok_start_position - doc_start + doc_offset
          end_position = tok_end_position - doc_start + doc_offset
      if self.is_training and self.is_impossible:
        start_position = 0
        end_position = 0
      
      self.input_tokens_list.append(tokens)
      self.input_ids_list.append(input_ids)
      self.input_mask_list.append([1] * len(input_ids))
      self.segment_ids_list.append(segment_ids)
      self.start_position_list.append(start_position)
      self.end_position_list.append(end_position)
      self.token_to_orig_map_list.append(token_to_orig_map)
      self.token_is_max_context_list.append(token_is_max_context)
      if self.is_training and not self.is_impossible:
        self.answer_list.append(tokens[start_position: end_position+1])
    self.max_span_length = max([len(input_ids) for input_ids in self.input_ids_list])
    self.sum_span_length = sum([len(input_ids) for input_ids in self.input_ids_list])
        
  def write_predictions(self, raw_features, raw_results, n_best_size, with_negative, max_answer_length=30):
    # We assume we can link all the raw_features and raw_result to this example
    # raw feature is ReadingComprehensionInputFeature
    # raw_results is Tuple(pred_start_logits, pred_end_logits)
    prelim_predictions = []
    score_null = 1000000 # large and positive
    min_null_feature_index = 0 # the paragraph slice with min null score
    null_start_logit = 0 # the start logit at the slice with min null score
    null_end_logit = 0 # the end logit at the slice with min null score
    if len(raw_features) != len(self.token_to_orig_map_list):
      for feature in raw_features:
        print(feature.guid)
        print(feature.input_ids)
      exit()
    for index, (feature, result) in enumerate(zip(raw_features, raw_results)):
      start_indexes = _get_best_indexes(result.start_logits, n_best_size)
      end_indexes = _get_best_indexes(result.end_logits, n_best_size)

      if with_negative:
        feature_null_score = result.start_logits[0] + result.end_logits[0]
        if feature_null_score < score_null:
          score_null = feature_null_score
          min_null_feature_index = index
          null_start_logit = result.start_logits[0]
          null_end_logit = result.end_logits[0]

      """TODO: need to fix the prediction filtering"""      
      for start_index in start_indexes:
        for end_index in end_indexes:
          """We are trying to throw all invalid predictions here"""
          if start_index >= len(feature.input_ids):
            continue
          if end_index >= len(feature.input_ids):
            continue
          if start_index not in self.token_to_orig_map_list[index]:
            continue
          if end_index not in self.token_to_orig_map_list[index]:
            continue
          if not self.token_is_max_context_list[index].get(start_index, False):
            continue
          if end_index < start_index:
            continue
          length = end_index - start_index + 1
          if length > max_answer_length:
            continue
          prelim_predictions.append(_PrelimPrediction(
            feature_index = index,
            start_index = start_index,
            end_index = end_index,
            start_logit = result.start_logits[start_index],
            end_logit = result.end_logits[end_index]))
    if with_negative:
      prelim_predictions.append(
        _PrelimPrediction(
          feature_index = min_null_feature_index,
          start_index = 0,
          end_index = 0,
          start_logit = null_start_logit,
          end_logit = null_end_logit))
    prelim_predictions = sorted(
      prelim_predictions,
      key=lambda x: (x.start_logit + x.end_logit),
      reverse=True)
    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
      if len(nbest) >= n_best_size:
        break
      if pred.start_index > 0: # this is a non-null prediction
        tok_tokens = self.input_tokens_list[pred.feature_index][pred.start_index:(pred.end_index + 1)]
        orig_doc_start = self.token_to_orig_map_list[pred.feature_index][pred.start_index]
        orig_doc_end = self.token_to_orig_map_list[pred.feature_index][pred.end_index]
        orig_tokens = self.raw_doc[orig_doc_start:(orig_doc_end + 1)]
        tok_text = " ".join(tok_tokens)

        # De-tokenize WordPieces that have been split off.
        tok_text = tok_text.replace(" ##", "")
        tok_text = tok_text.replace("##", "")

        # Clean whitespace
        tok_text = tok_text.strip()
        tok_text = " ".join(tok_text.split())
        orig_text = " ".join(orig_tokens)

        final_text = get_final_text(tok_text, orig_text, do_lower_case=True, verbose_logging=True)
        if final_text in seen_predictions:
          continue

        seen_predictions[final_text] = True
      else:
        final_text = ""
        seen_predictions[final_text] = True
      nbest.append(
        _NbestPrediction(
          text=final_text,
          start_logit=pred.start_logit,
          end_logit=pred.end_logit))
    if with_negative:
      if "" not in seen_predictions:
        nbest.append(
          _NbestPrediction(
            text="",
            start_logit=null_start_logit,
            end_logit=null_end_logit))

      # In very rare edge cases we could only have single null prediction.
      # So we just create a nonce prediction in this case to avoid failure.
      if len(nbest) == 1:
        nbest.insert(0,
                     _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
      nbest.append(
        _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

    assert len(nbest) >= 1

    total_scores = []
    best_non_null_entry = None
    best_non_null_entry_idx = -1
    for idx, entry in enumerate(nbest):
      total_scores.append(entry.start_logit + entry.end_logit)
      if not best_non_null_entry:
        if entry.text:
          best_non_null_entry = entry
          best_non_null_entry_idx = idx


    probs = _compute_softmax(total_scores)
    best_non_null_entry_score = probs[best_non_null_entry_idx]

    nbest_json = []
    for (i, entry) in enumerate(nbest):
      output = collections.OrderedDict()
      output["text"] = entry.text
      output["probability"] = probs[i]
      output["start_logit"] = entry.start_logit
      output["end_logit"] = entry.end_logit
      nbest_json.append(output)

    assert len(nbest_json) >= 1

    if not with_negative:
      self.prediction = nbest_json[0]["text"]
    else:
      # predict "" iff the null score - the score of best non-null > threshold
      score_diff = score_null - best_non_null_entry.start_logit - (
        best_non_null_entry.end_logit)
      self.scores_diff = score_diff.item()
      # if score_diff > null_score_diff_threshold:
      #   self.prediction = ""
      #   self.span_score = 0
      # else:
      #   self.prediction = best_non_null_entry.text
      #   self.span_score = best_non_null_entry_score
      self.prediction = best_non_null_entry.text
      self.span_score = best_non_null_entry_score
      self.nbest_json = nbest_json

  @property
  def len(self):
    return self.sum_span_length

  @property
  def max_len(self):
    return self.max_span_length

class ReadingComprehensionInputFeature(object):
  """TODO: need to fix the feature design"""
  def __init__(self,
               guid,
               input_ids,
               input_mask,
               segment_ids,
               label_ids,
               is_impossible):
    self.guid = guid
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_ids = label_ids
    self.is_impossible = is_impossible

def get_reading_comprehension_examples(path):
  examples = []
  with open(path, "r", encoding='utf-8') as fin:
    for line in fin:
      example = json.loads(line)
      examples.append(ReadingComprehensionExample(
        guid=example['guid'],
        text_a=example['question_text'],
        text_b=" ".join(example['doc_tokens']),
        start_position=example['start_position'],
        end_position=example['end_position'],
        is_impossible=example['is_impossible'],
        orig_answer_text=example['orig_answer_text']))
  return examples

def convert_reading_comprehension_examples_to_features(examples, max_seq_length, extra_args=None):
  features = []
  max_length = max([example.max_len for example in examples])
  for idx, example in enumerate(examples):
    if not example.is_valid:
      continue
    for _input_ids, _input_mask, _segment_ids, _start_position, _end_position in zip(
        example.input_ids_list, example.input_mask_list, 
        example.segment_ids_list, example.start_position_list, example.end_position_list):
      padding = [0] * (max_length - len(_input_ids))
      input_ids = _input_ids + padding
      input_mask = _input_mask + padding
      segment_ids = _segment_ids + padding
      label_ids = [_start_position, _end_position]

      features.append(
        ReadingComprehensionInputFeature(
          guid=example.guid,
          input_ids = input_ids,
          input_mask = input_mask,
          segment_ids = segment_ids,
          label_ids = label_ids,
          is_impossible=example.is_impossible))
      
  return features

if __name__ == "__main__":
  from tokenizer.tokenization import BertTokenizer
  tokenizer = BertTokenizer.from_pretrained(
      "vocabs/tacred-bert-base-cased-vocab.txt", do_lower_case=False)
  # context = "The leader was John Smith (1895-1943)."
  # answer = "1895"
  # token, is_head = tokenizer.tokenize(context)
  # print(token, is_head)
  # start = 27
  # end = 30
  extra_args = {
    "max_query_length": 64,
    "max_seq_length": 384,
    "doc_stride": 128
  }
  examples = get_reading_comprehension_examples("data/raw_data/squad11/train.json")
  counter = 0
  for example in examples:
    example.process(tokenizer, extra_args)
    counter += len(example.input_ids_list)
  print(counter)
