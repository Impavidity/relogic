from relogic.logickit.inference.modeling import BertPreTrainedModel, BertModel
from relogic.logickit.inference.modeling_xlm import XLMPreTrainedModel, XLMModel
from relogic.logickit.modules.contextualizers.highway_lstm import HighwayLSTM
from relogic.utils.file_utils import cached_path
import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)

def get_encoder(encoder_type):
  if encoder_type == "bert":
    return Encoder
  if encoder_type == "xlm":
    return XLMEncoder
  if encoder_type == "xlmr":
    return XLMRobertaEncoder
  if encoder_type == "lstm":
    return LSTMEncoder
  if encoder_type == "embedding":
    return PretrainedWordEmbedding

PRETRAINED_VECTOR_ARCHIVE_MAP = {
  "fasttext-en": "https://git.uwaterloo.ca/p8shi/data-server/raw/master/embeddings/wiki-news-300d-1M.npy"
}

class PretrainedWordEmbedding(nn.Module):
  def __init__(self, pretrained_model_name_or_path, embedding_file_path):
    super().__init__()
    embedding_data = torch.from_numpy(np.load(embedding_file_path))
    self.word_embedding = nn.Embedding(embedding_data.size(0), embedding_data.size(1))
    self.word_embedding.weight.data.copy_(embedding_data)
    self.word_embedding.weight.requires_grad = False
    del embedding_data
    print("Loadding embedding from {}".format(embedding_file_path))

  def forward(self, *inputs, **kwargs):
    input_token_ids = kwargs.pop("_input_token_ids")
    input_embedding = self.word_embedding(input_token_ids)
    return input_embedding

  @classmethod
  def from_pretrained(cls, pretrained_model_name_or_path, cache_dir=None, output_attentions=False):
    if pretrained_model_name_or_path in PRETRAINED_VECTOR_ARCHIVE_MAP:
      embedding_file = PRETRAINED_VECTOR_ARCHIVE_MAP[pretrained_model_name_or_path]
    else:
      embedding_file = pretrained_model_name_or_path
    try:
      resolved_embedding_file = cached_path(embedding_file, cache_dir=cache_dir)
    except EnvironmentError:
      logger.error(
        "Model name '{}' was not found in model name list ({}). "
        "We assumed '{}' was a path or url but couldn't find any file "
        "associated to this path or url.".format(
          pretrained_model_name_or_path,
          ', '.join(PRETRAINED_VECTOR_ARCHIVE_MAP.keys()),
          embedding_file))
      return None
    if resolved_embedding_file == embedding_file:
      logger.info("will load embedding file from {}".format(embedding_file))
    else:
      logger.info("will load embedding file {} from cache at {}".format(
        embedding_file, resolved_embedding_file))
    return cls(pretrained_model_name_or_path, embedding_file_path=resolved_embedding_file)


class CNNEncoder(nn.Module):
  def __init__(self, pretrained_model_name_or_path, embedding_file_path):
    super().__init__()
    self.encoder = None
    embedding_data = torch.from_numpy(np.load(embedding_file_path))
    self.word_embedding = nn.Embedding(embedding_data.size(0), embedding_data.size(1))
    self.word_embedding.weight.data.copy_(embedding_data)
    del embedding_data
    print("Loading embedding from {}".format(embedding_file_path))

  def forward(self, *input, **kwargs):
    pass

  @classmethod
  def from_pretrained(cls, pretrained_model_name_or_path, cache_dir=None, output_attentions=False):
    if pretrained_model_name_or_path in PRETRAINED_VECTOR_ARCHIVE_MAP:
      embedding_file = PRETRAINED_VECTOR_ARCHIVE_MAP[pretrained_model_name_or_path]
    else:
      embedding_file = pretrained_model_name_or_path
    try:
      resolved_embedding_file = cached_path(embedding_file, cache_dir=cache_dir)
    except EnvironmentError:
      logger.error(
        "Model name '{}' was not found in model name list ({}). "
        "We assumed '{}' was a path or url but couldn't find any file "
        "associated to this path or url.".format(
          pretrained_model_name_or_path,
          ', '.join(PRETRAINED_VECTOR_ARCHIVE_MAP.keys()),
          embedding_file))
      return None
    if resolved_embedding_file == embedding_file:
      logger.info("will load embedding file from {}".format(embedding_file))
    else:
      logger.info("will load embedding file {} from cache at {}".format(
        embedding_file, resolved_embedding_file))
    return cls(pretrained_model_name_or_path, embedding_file_path=resolved_embedding_file)


class LSTMEncoder(nn.Module):
  def __init__(self, pretrained_model_name_or_path, embedding_file_path):
    super().__init__()
    self.encoder = HighwayLSTM(num_layers=2, input_size=300, hidden_size=200, layer_dropout=0.2)
    embedding_data = torch.from_numpy(np.load(embedding_file_path))
    self.word_embedding = nn.Embedding(embedding_data.size(0), embedding_data.size(1))
    self.word_embedding.weight.data.copy_(embedding_data)
    del embedding_data
    print("Loading embedding from {}".format(embedding_file_path))

  def forward(self, *inputs, **kwargs):
    input_token_ids = kwargs.pop("_input_token_ids")
    token_length = kwargs.pop("_token_length")
    input_embedding = self.word_embedding(input_token_ids)
    sequence_output = self.encoder(inputs=input_embedding, lengths=token_length)
    return sequence_output


  @classmethod
  def from_pretrained(cls, pretrained_model_name_or_path, cache_dir=None, output_attentions=False):
    if pretrained_model_name_or_path in PRETRAINED_VECTOR_ARCHIVE_MAP:
      embedding_file = PRETRAINED_VECTOR_ARCHIVE_MAP[pretrained_model_name_or_path]
    else:
      embedding_file = pretrained_model_name_or_path
    try:
      resolved_embedding_file = cached_path(embedding_file, cache_dir=cache_dir)
    except EnvironmentError:
      logger.error(
        "Model name '{}' was not found in model name list ({}). "
        "We assumed '{}' was a path or url but couldn't find any file "
        "associated to this path or url.".format(
          pretrained_model_name_or_path,
          ', '.join(PRETRAINED_VECTOR_ARCHIVE_MAP.keys()),
          embedding_file))
      return None
    if resolved_embedding_file == embedding_file:
      logger.info("will load embedding file from {}".format(embedding_file))
    else:
      logger.info("will load embedding file {} from cache at {}".format(
        embedding_file, resolved_embedding_file))
    return cls(pretrained_model_name_or_path, embedding_file_path=resolved_embedding_file)

class XLMRobertaEncoder(nn.Module):
  def __init__(self, pretrained_model_name_or_path):
    super().__init__()
    self.xlmr = torch.hub.load('pytorch/fairseq', pretrained_model_name_or_path)

  def forward(self, input_ids, **kwargs):
    sequence_output = self.xlmr.extract_features(input_ids)
    return sequence_output

  @classmethod
  def from_pretrained(cls, pretrained_model_name_or_path, output_attentions=False):
    assert(not output_attentions)
    # Need to figure out how to output attention from XLMR
    return cls(pretrained_model_name_or_path)


class XLMEncoder(nn.Module):
  def __init__(self, pretrained_model_name_or_path):
    super(XLMEncoder, self).__init__()
    self.xlm = XLMModel.from_pretrained(
      pretrained_model_name_or_path)

  def forward(self,
              input_ids,
              attention_mask=None,
              langs=None,
              token_type_ids=None,
              position_ids=None,
              lengths=None,
              cache=None,
              head_mask=None,
              output_all_encoder_layers=False,
              selected_non_final_layers=None,
              route_path=None,
              no_dropout=True,
              **kwargs):
    """

    :param input_ids:
    :param attention_mask:
    :param langs: (batch_size, sentence_length)
    :param token_type_ids:
    :param position_ids:
    :param lengths:
    :param cache:
    :param head_mask:
    :param output_all_encoder_layers:
    :param selected_non_final_layers:
    :param route_path:
    :param no_dropout:
    :return:
    """
    sequence_output = self.xlm(input_ids=input_ids,
             attention_mask=attention_mask,
             langs=langs,
             token_type_ids=token_type_ids,
             position_ids=position_ids,
             lengths=lengths,
             cache=cache,
             head_mask=head_mask)
    return sequence_output

  @classmethod
  def from_pretrained(cls, pretrained_model_name_or_path, output_attentions=False):
    assert(not output_attentions)
    # TODO: Figure out how to output attention map from XLM
    return cls(pretrained_model_name_or_path)




class Encoder(BertPreTrainedModel):
  def __init__(self, config, output_attentions=False):
    super(Encoder, self).__init__(config)
    self._config = config
    self.output_attentions = output_attentions
    self.bert = BertModel(config, output_attentions=output_attentions)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.apply(self.init_bert_weights)

  def forward(self,
              input_ids,
              token_type_ids=None,
              attention_mask=None,
              head_mask=None,
              output_all_encoded_layers=False,
              token_level_attention_mask=None,
              selected_non_final_layers=None,
              route_path=None,
              no_dropout=False,
              **kwargs):
    sequence_output = self.bert(
      input_ids=input_ids,
      token_type_ids=token_type_ids,
      attention_mask=attention_mask,
      head_mask=head_mask,
      token_level_attention_mask=token_level_attention_mask,
      output_all_encoded_layers=output_all_encoded_layers,
      selected_non_final_layers=selected_non_final_layers)
    if self.output_attentions:
      attention_map, sequence_output, _ = sequence_output
    else:
      sequence_output, _ = sequence_output
    if not no_dropout:
      if output_all_encoded_layers or selected_non_final_layers is not None:
        sequence_output = [self.dropout(seq) for seq in sequence_output]
      else:
        sequence_output = self.dropout(sequence_output)
    if self.output_attentions:
      return sequence_output, attention_map
    else:
      return sequence_output


if __name__ == "__main__":
  import torch
  # model = BertModel.from_pretrained("bert-large-uncased-whole-word-masking")
  model = LSTMEncoder.from_pretrained("fasttext-en")
  # layer1 = model.encoder.layer[0]
  # print(layer1[0].attention.self.value.weight.data - layer1[1].attention.self.value.weight.data)
  feature = model(_input_token_ids=torch.zeros(2, 5, dtype=torch.long),
                 _token_length=torch.LongTensor([3, 5]))

  print(feature)

  # encoder = XLMEncoder("xlm-mlm-tlm-xnli15-1024")
  # feature = encoder(input_ids=torch.zeros(1, 5, dtype=torch.long))
  # print(feature)
