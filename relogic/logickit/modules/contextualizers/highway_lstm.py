import torch.nn as nn
import torch
from relogic.logickit.modules.initializers import block_orthogonal
from relogic.logickit.modules.rnn import dynamic_rnn

class LSTM(nn.Module):
  """
  Args:
    input_size: The number of expected features in the input `x`
    hidden_size: The number of features in the hidden state `h`
    num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
        would mean stacking two LSTMs together to form a `stacked LSTM`,
        with the second LSTM taking in outputs of the first LSTM and
        computing the final results. Default: 1
    bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
        Default: ``True``
    batch_first: If ``True``, then the input and output tensors are provided
        as (batch, seq, feature). Default: ``False``
    dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
        LSTM layer except the last layer, with dropout probability equal to
        :attr:`dropout`. Default: 0
    bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``

  Attributes:
    weight_ih_l[k] : the learnable input-hidden weights of the :math:`\text{k}^{th}` layer
        `(W_ii|W_if|W_ig|W_io)`, of shape `(4*hidden_size, input_size)` for `k = 0`.
        Otherwise, the shape is `(4*hidden_size, num_directions * hidden_size)`
    weight_hh_l[k] : the learnable hidden-hidden weights of the :math:`\text{k}^{th}` layer
        `(W_hi|W_hf|W_hg|W_ho)`, of shape `(4*hidden_size, hidden_size)`
    bias_ih_l[k] : the learnable input-hidden bias of the :math:`\text{k}^{th}` layer
        `(b_ii|b_if|b_ig|b_io)`, of shape `(4*hidden_size)`
    bias_hh_l[k] : the learnable hidden-hidden bias of the :math:`\text{k}^{th}` layer
        `(b_hi|b_hf|b_hg|b_ho)`, of shape `(4*hidden_size)`

  """
  def __init__(self,
               input_size,
               hidden_size,
               num_layers=1,
               bias=True,
               batch_first=False,
               dropout=0,
               bidirectional=False):
    super(LSTM, self).__init__()
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
      bias, batch_first, dropout, bidirectional)
    for layer in self.lstm._all_weights:
      for param in layer:
        if 'weight_ih' in param:
          block_orthogonal(
            tensor=self.lstm.__getattr__(param),
            split_sizes=[hidden_size, input_size])
        if 'weight_hh' in param:
          block_orthogonal(
            tensor=self.lstm.__getattr__(param),
            split_sizes=[hidden_size, hidden_size])

  def forward(self, inputs, lengths=None):
    if lengths is None:
      return self.lstm(inputs)
    else:
      return dynamic_rnn(self.lstm, inputs, lengths)

class HighwayLSTM(nn.Module):
  def __init__(self, num_layers, input_size, hidden_size, layer_dropout):
    super(HighwayLSTM, self).__init__()
    self.num_layers = num_layers
    self.layers = nn.ModuleList([LSTM(
      input_size=input_size if layer == 0 else hidden_size * 2,  # for indicator embedding
      hidden_size=hidden_size,
      bidirectional=True,
      batch_first=True,
      num_layers=1
    ) for layer in range(num_layers)])
    self.dropout = nn.Dropout(p=layer_dropout)
    self.linear_layers = nn.ModuleList([torch.nn.Linear(hidden_size * 2, hidden_size * 2)
                                        for _ in range(num_layers - 1)])


  def forward(self, inputs, lengths):
    current_inputs = inputs
    for idx, layer in enumerate(self.layers):
      output, _ = layer(current_inputs, lengths)
      output = self.dropout(output)
      if idx > 0:
        gate = self.linear_layers[idx-1](output)
        gate = torch.sigmoid(gate)
        output = gate * output + (1 - gate) * current_inputs
      current_inputs = output
    return output