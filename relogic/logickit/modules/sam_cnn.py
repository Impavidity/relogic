import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
  def __init__(self, num_of_conv, in_channels, out_channels, kernel_size, in_features, out_features=None, stride=1,
               dilation=1, groups=1, bias=True, active_func=F.relu, pooling=F.max_pool1d,
               dropout=0.5, padding_strategy="default", padding_list=None, fc_layer=True,
               include_map=False, k_max_pooling=False, k=1):
    """
    :param num_of_conv: Follow kim cnn idea
    :param kernel_size: if is int type, then make it into list, length equals to num_of_conv
                 if list type, then check the length of it, should has length of num_of_conv
    :param out_features: feature size
    """
    super(SimpleCNN, self).__init__()
    if type(kernel_size) == int:
      kernel_size = [kernel_size]
    if len(kernel_size) != num_of_conv:
      print("Number of kernel_size should be same num_of_conv")
      exit(1)
    if padding_list == None:
      if padding_strategy == "default":
        padding_list = [(k_size - 1, 0) for k_size in kernel_size]
    self.include_map = include_map

    self.conv = nn.ModuleList([nn.Conv2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=(k_size, in_features),
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=groups,
                                         bias=bias)
                               for k_size, padding in zip(kernel_size, padding_list)])
    self.pooling = pooling
    self.k_max_pooling = k_max_pooling
    self.k = k if k_max_pooling else 1
    self.active_func = active_func
    self.fc_layer = fc_layer
    if fc_layer:
      self.dropout = nn.Dropout(dropout)
      self.fc = nn.Linear(num_of_conv * out_channels * self.k, out_features)

  def kmax_pooling(self, x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)

  def forward(self, input):
    batch_size = input.size(0)
    if len(input.size()) == 3:
      input = input.unsqueeze(1)
    # input = (batch, in_channels, sent_len, word_dim)
    x_map = [self.active_func(conv(input)).squeeze(3) for conv in self.conv]
    # (batch, channel_output, ~=sent_len) * Ks
    if self.k_max_pooling:
      x = [self.kmax_pooling(i, 2, self.k).view(batch_size, -1) for i in x_map]
    else:
      x = [self.pooling(i, i.size(2)).squeeze(2) for i in x_map]  # max-over-time pooling
    x = torch.cat(x, 1)  # (batch, out_channels * Ks)
    if self.fc_layer:
      x = self.dropout(x)
      x = self.fc(x)
    if self.include_map == False:
      return x
    else:
      return x, x_map

class SamCNN(nn.Module):
  def __init__(self, config, task_name, n_classes):
    super().__init__()

    self.sm_cnn = SimpleCNN(
      num_of_conv=1,
      in_channels=1,
      out_channels=config.output_channel,
      kernel_size=[config.kernel_size],
      in_features=config.word_embed_dim,
      out_features=config.hidden_size,
      active_func=nn.ReLU(),
      dropout=config.dropout,
      fc_layer=True
    )

    self.projection = nn.Linear(config.hidden_size * 3,
                config.projection_size)
    self.batch_norm = nn.BatchNorm1d(config.projection_size)
    self.to_logits = nn.Sequential(
      nn.Tanh(),
      nn.Dropout(config.dropout),
      nn.Linear(config.projection_size, n_classes))

    self.context_cnn = SimpleCNN(
      num_of_conv=1,
      in_channels=1,
      out_channels=config.output_channel,
      kernel_size=[config.kernel_size],
      in_features=config.word_embed_dim,
      out_features=config.hidden_size,
      active_func=nn.ReLU(),
      dropout=config.dropout,
      fc_layer=True
    )
    self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

  def attn_context_ave(self, question_embed, answer_embed, batch_size):
    question_len = question_embed.size(1)
    answer_len = answer_embed.size(1)
    dimension = question_embed.size(2)
    question = torch.cat([question_embed.unsqueeze(2)] * answer_len, dim=2).view(-1, dimension)
    answer = torch.cat([answer_embed.unsqueeze(1)] * question_len, dim=1).view(-1, dimension)
    # (batch, question_len, answer_len, dim)
    attn_prob = self.cos(answer, question).unsqueeze(1)
    attn_answer = (answer * attn_prob).view(batch_size * question_len, answer_len, dimension)
    feature = self.context_cnn(attn_answer).view(batch_size, question_len, -1)
    feature = torch.sum(feature, dim=1) / question_len
    return feature

  def forward(self, *inputs, **kwargs):
    a_features = kwargs.pop("a_features")
    b_features = kwargs.pop("b_features")

    a_cnn_features = self.sm_cnn(a_features)
    b_cnn_features = self.sm_cnn(b_features)
    attention_features = self.attn_context_ave(a_features, b_features, a_features.size(0))

    feat_comb = torch.cat([a_cnn_features, b_cnn_features, attention_features], dim=1)
    feat_comb = self.projection(feat_comb)
    if feat_comb.size(0) > 1:
      feat_comb = self.batch_norm(feat_comb)
    logits = self.to_logits(feat_comb)

    return logits



