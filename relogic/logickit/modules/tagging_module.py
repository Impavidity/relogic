import torch.nn as nn
import copy
import torch
from relogic.logickit.modules.rnn import dynamic_rnn


# class PredictionModule(nn.Module):
#   def __init__(self, config, task_name, n_classes, activate=True):
#     super(PredictionModule, self).__init__()
#     self.config = config
#     self.task_name = task_name
#     self.n_classes = n_classes
#     # self.projection = nn.Linear(config.hidden_size, config.projection_size)
#     # self.activate = activate
#     # if activate:
#     #   self.activation = nn.ReLU()
#     self.to_logits = nn.Linear(config.hidden_size, self.n_classes)
#     self.apply(self.init_weights)
#
#   def init_weights(self, module):
#     if isinstance(module, nn.Linear):
#       module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#     if isinstance(module, nn.Linear) and module.bias is not None:
#       module.bias.data.zero_()
#
#   def forward(self, input):
#     # projected = self.projection(input)
#     # if self.activate:
#     #   projected = self.activation(projected)
#     logits = self.to_logits(input)
#     return logits

class PredictionModule(nn.Module):
  def __init__(self, config, task_name, n_classes, activate=True):
    super(PredictionModule, self).__init__()
    self.config = config
    self.task_name = task_name
    self.n_classes = n_classes
    if self.config.use_bilstm:
      self.projection = nn.Linear(2 * config.hidden_size, config.hidden_size)
      self.activate = activate
      if activate:
        self.activation = nn.SELU()
    self.to_logits = nn.Linear(config.hidden_size, self.n_classes)
    # self.apply(self.init_weights)

  def init_weights(self, module):
    if isinstance(module, nn.Linear):
      module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    if isinstance(module, nn.Linear) and module.bias is not None:
      module.bias.data.zero_()

  def forward(self, input):
    if self.config.use_bilstm:
      projected = self.projection(input)
      if self.activate:
        projected = self.activation(projected)
    else:
      projected = input
    logits = self.to_logits(projected)
    return logits

class PartialViewPredictionModule(nn.Module):
  def __init__(self, config, task_name, n_classes, activate=True):
    super(PartialViewPredictionModule, self).__init__()
    self.config = config

    self.projection = nn.Linear(config.hidden_size, config.hidden_size)
    self.activate = activate
    if activate:
      self.activation = nn.SELU()
    self.to_logits = nn.Linear(config.hidden_size, n_classes)
    # self.apply(self.init_weights)

  def init_weights(self, module):
    if isinstance(module, nn.Linear):
      module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    if isinstance(module, nn.Linear) and module.bias is not None:
      module.bias.data.zero_()

  def forward(self, input):
    projected = self.projection(input)
    if self.activate:
      projected = self.activation(projected)
    logits = self.to_logits(projected)
    return logits

class TaggingModule(nn.Module):
  def __init__(self, config, task_name, n_classes):
    super(TaggingModule, self).__init__()
    self.config = config
    self.partial_view_sources = config.partial_view_sources
    if self.config.use_bilstm:
      self.bilstm = nn.LSTM(
        input_size=config.hidden_size,  # for indicator embedding
        hidden_size=config.hidden_size,
        bidirectional=True,
        batch_first=True,
        num_layers=1)
    self.primary = PredictionModule(config, task_name, n_classes)
    # self.partial_list = nn.ModuleList([
    #   PredictionModule(config, task_name, n_classes)
    #   for _ in range(len(config.partial_view_sources))])
    if self.config.is_semisup:
      self.partial_list = nn.ModuleList([
        PartialViewPredictionModule(config, task_name, n_classes)
        for _ in range(4)])

  def padding(self, batch, dim, device):
    return torch.zeros([batch, 1, dim], device=device).float()

  def forward(self,
              input,
              view_type="primary",
              final_multi_head_repr=None,
              input_mask=None,
              segment_ids=None,
              extra_args=None):
    # print(input_mask)
    input_lengths = (input_mask == 1).sum(-1)
    if self.config.use_bilstm:
      output, (ht, ct) = dynamic_rnn(self.bilstm, input, input_lengths)
    else:
      output = input
    if view_type == "primary":
      primary_logits = self.primary(output)
      return primary_logits
    else:
      # if final_multi_head_repr is None:
      #   raise ValueError("final_multi_head_repr should have value instead of None")
      # new_shape_final_multi_head_repr = final_multi_head_repr.size()[:-1] + \
      #    (int(final_multi_head_repr.size(-1) / self.config.self_attention_head_size), self.config.self_attention_head_size, )
      # final_multi_head_repr = final_multi_head_repr.view(*new_shape_final_multi_head_repr)
      partial_view_results = []
      padding = self.padding(output.size(0), self.config.hidden_size, output.device)
      for idx, partial_view_module in enumerate(self.partial_list):
        # partial_view_results.append(partial_view_module(input[source_layer]))
        # partial_view_results.append(partial_view_module(final_multi_head_repr[:, :, head_index, :]))
        if idx == 0:
          partial_view_results.append(partial_view_module(output[:,:,:self.config.hidden_size]))
        if idx == 1:
          partial_view_results.append(partial_view_module(output[:,:,self.config.hidden_size:]))
        if idx == 2:
          partial_view_results.append(
            partial_view_module(torch.cat([padding, output[:,:-1,:self.config.hidden_size]], dim=1)))
        if idx == 3:
          partial_view_results.append(
            partial_view_module(torch.cat([output[:,1:,self.config.hidden_size:], padding], dim=1)))
      return partial_view_results