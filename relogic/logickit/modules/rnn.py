import torch
from torch import nn


def dynamic_rnn(rnn, inputs, seq_lengths):
  """
  :param rnn: RNN instance
  :param inputs: FloatTensor, shape [batch, time, dim] if rnn.batch_first else [time, batch, dim]
  :param seq_lengths: LongTensor shape [batch]
  :param hidden_state: FloatTensor,
  :return: the result of rnn layer
  """
  batch_first = rnn.batch_first
  sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=True)
  _, desorted_indices = torch.sort(indices, descending=False)
  if batch_first:
    inputs = inputs[indices]
  else:
    inputs = inputs[:, indices]
  packed_inputs = nn.utils.rnn.pack_padded_sequence(
    inputs,
    sorted_seq_lengths.cpu().numpy(),
    batch_first=batch_first)
  res, states = rnn(packed_inputs)
  padded_res, _ = nn.utils.rnn.pad_packed_sequence(res, batch_first=batch_first)
  if batch_first:
    desorted_res = padded_res[desorted_indices]
  else:
    desorted_res = padded_res[:, desorted_indices]
  if isinstance(states, tuple):
    states =  tuple(state[:, desorted_indices] for state in states)
  else:
    states = states[:, desorted_indices]
  return desorted_res, states