from relogic.logickit.inference.modeling import BertPreTrainedModel, BertModel
from relogic.logickit.inference.modeling_xlm import XLMPreTrainedModel, XLMModel
from relogic.logickit.modules.contextualizers.highway_lstm import HighwayLSTM
from relogic.utils.file_utils import cached_path
import torch
import torch.nn as nn
import numpy as np
import logging
from relogic.logickit.modules.span_extractors.average_span_extractor import AverageSpanExtractor

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
  "fasttext-en": "https://git.uwaterloo.ca/p8shi/data-server/raw/master/embeddings/wiki-news-300d-1M.npy",
  "glove-300d-6B": "https://git.uwaterloo.ca/p8shi/data-server/raw/master/embeddings/glove-300d-6B.npy",
  "glove-300d-42B": "https://git.uwaterloo.ca/p8shi/data-server/raw/master/embeddings/glove-300d-42B.npy",
}


class PretrainedWordEmbedding(nn.Module):
  def __init__(self, pretrained_model_name_or_path, embedding_file_path):
    super().__init__()
    embedding_data = torch.from_numpy(np.load(embedding_file_path))
    # self.word_embedding = nn.Embedding.from_pretrained(embedding_data)
    self.word_embedding = nn.Embedding(embedding_data.size(0), embedding_data.size(1))
    self.word_embedding.weight.data.copy_(embedding_data)
    # self.word_embedding.weight.requires_grad = False
    del embedding_data
    print("Loadding embedding from {}".format(embedding_file_path))
    self.average_span_extractor = AverageSpanExtractor()


  def set_requires_grad(self, mode, dictionary=None):
    """
    The mode choices: all_false, partial_false, partial_true, all_true
    The dictionary is used for partial_false and partial_true.
    If the mode is partial_false, the ids in the dictionary file will be used to set
      the corresponding vector's requires_grad as False. And vice versa.
    The format of the file:
      id\tword_text\n
    or
      id\n
    Did some search, I find two solutions
    https://stackoverflow.com/questions/54924582/is-it-possible-to-freeze-only-certain-embedding-weights-in-the-embedding-layer-i
    https://discuss.pytorch.org/t/updating-part-of-an-embedding-matrix-only-for-out-of-vocab-words/33297/4
    I will take the second PartiallyFixedEmbedding in the second post.
    And it did the zero grad on the whole embedding. However, I will create the mask for each inputs.
    """
    self.mode = mode
    if mode == "all_false":
      self.word_embedding.weight.requires_grad = False
    elif mode == "all_true":
      self.word_embedding.weight.requires_grad = True
    else:
      raise ValueError(f"Known mode {mode}")


  def forward(self, *inputs, **kwargs):
    input_token_ids = kwargs.pop("_input_token_ids")
    input_token_ids_mask = kwargs.pop("_input_token_ids_mask", None)

    input_embedding = self.word_embedding(input_token_ids)
    if input_token_ids_mask is not None:
      def zero_grad_fixed(gr):
        return gr * input_token_ids_mask
      input_embedding.register_hook(zero_grad_fixed)

    level = kwargs.pop("aggregation_level", "word")  # word or span
    if level == "span":
      input_token_spans = kwargs.pop("_input_token_spans")
      input_token_span_masks = kwargs.pop("_input_token_span_masks")
      # do the span average
      averaged_span_embedding = self.average_span_extractor(
        sequence_tensor=input_embedding,
        span_indices=input_token_spans,
        span_indices_mask=input_token_span_masks)
      return averaged_span_embedding

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
  def __init__(self, word_embedding_module):
    super().__init__()
    self.word_embedding_module = word_embedding_module
    self.encoder = None


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

    embedding_data = torch.from_numpy(np.load(resolved_embedding_file))
    word_embedding = nn.Embedding(embedding_data.size(0), embedding_data.size(1))
    word_embedding.weight.data.copy_(embedding_data)
    del embedding_data
    print("Loading embedding from {}".format(resolved_embedding_file))
    return cls(word_embedding_module=word_embedding)


class LSTMEncoder(nn.Module):
  def __init__(self, word_embedding_module, num_layers=1, input_size=300, hidden_size=128, layer_dropout=0.2):
    super().__init__()
    self.word_embedding = word_embedding_module
    self.encoder = HighwayLSTM(num_layers=num_layers, input_size=input_size, hidden_size=hidden_size, layer_dropout=layer_dropout)

  def forward(self, *inputs, **kwargs):
    input_token_ids = kwargs.pop("_input_token_ids")
    input_token_ids_mask = kwargs.pop("_input_token_ids_mask", None)
    input_token_length = kwargs.pop("_input_token_length", None)
    input_token_spans = kwargs.pop("_input_token_spans", None)
    input_token_span_masks = kwargs.pop("_input_token_span_masks", None)
    aggregation_level = kwargs.pop("aggregation_level", "word")
    input_embedding = self.word_embedding(
      _input_token_ids=input_token_ids,
      _input_token_ids_mask=input_token_ids_mask,
      _input_token_spans=input_token_spans,
      _input_token_span_masks=input_token_span_masks,
      aggregation_level=aggregation_level)
    if aggregation_level == "span":
      input_token_length = input_token_span_masks.sum(-1)
      # Change the input_token_length to input_span_lengths
    assert input_token_length is not None
    sequence_output, state = self.encoder(inputs=input_embedding, lengths=input_token_length)
    return sequence_output, state, input_embedding


  @classmethod
  def from_pretrained(cls, pretrained_model_name_or_path, cache_dir=None, output_attentions=False, **kwargs):
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

    embedding_data = torch.from_numpy(np.load(resolved_embedding_file))
    word_embedding = nn.Embedding(embedding_data.size(0), embedding_data.size(1))
    word_embedding.weight.data.copy_(embedding_data)
    del embedding_data
    print("Loading embedding from {}".format(resolved_embedding_file))

    return cls(word_embedding_module=word_embedding, **kwargs)

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
