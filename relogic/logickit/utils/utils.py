import numpy as np
import torch

def softmax(x):
  """Compute softmax values for each sets of scores in x."""
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=-1)

def gen_position_indicator(span, length):
  indicator = [0] * length
  for idx, i in enumerate(range(span[0], -1, -1)):
    indicator[i] = -idx
  for idx, i in enumerate(range(span[1], length)):
    indicator[i] = idx + 1
  return indicator

def indicator_vector(index, length, default_label=0, indicator_label=1, head_index=None):
  vector = [default_label] * length
  if head_index is None:
    for idx in index:
      vector[idx] = indicator_label
  else:
    for idx in index:
      vector[head_index[idx]] = indicator_label
  return vector

def truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

def get_span_labels(sentence_tags, is_head=None, segment_id=None, inv_label_mapping=None, ignore_label=list(["V"])):
  """Go from token-level labels to list of entities (start, end, class)."""
  if inv_label_mapping:
    sentence_tags = [inv_label_mapping[i] for i in sentence_tags]
  filtered_sentence_tag = []
  if is_head:
    # assert(len(sentence_tags) == len(is_head))

    for idx, (head, segment) in enumerate(zip(is_head, segment_id)):
      if head == 1 and segment == 0:
        if sentence_tags[idx] != 'X':
          filtered_sentence_tag.append(sentence_tags[idx])
        else:
          filtered_sentence_tag.append("O")
  if filtered_sentence_tag:
    sentence_tags = filtered_sentence_tag
  span_labels = []
  last = 'O'
  start = -1
  for i, tag in enumerate(sentence_tags):
    pos, _ = (None, 'O') if tag == 'O' else tag.split('-', 1)
    if (pos == 'S' or pos == 'B' or tag == 'O') and last != 'O':
      span_labels.append((start, i - 1, last.split('-')[-1]))
    if pos == 'B' or pos == 'S' or last == 'O':
      start = i
    last = tag
  if sentence_tags[-1] != 'O':
    span_labels.append((start, len(sentence_tags) - 1,
                        sentence_tags[-1].split('-', 1)[-1]))
  for item in span_labels:
    if item[2] in ignore_label:
      span_labels.remove(item)
  return set(span_labels), sentence_tags

def filter_head_prediction(sentence_tags, is_head):
  filtered_sentence_tag = []
  for idx, head in enumerate(is_head):
    if head == 1:
      if sentence_tags[idx] != 'X':
        filtered_sentence_tag.append(sentence_tags[idx])
      else:
        filtered_sentence_tag.append("O")
  return filtered_sentence_tag

def create_tensor(features, attribute, dtype, device):
  return torch.tensor([getattr(f, attribute) for f in features], dtype=dtype).to(device)