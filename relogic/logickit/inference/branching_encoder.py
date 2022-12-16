from relogic.logickit.inference.modeling import PRETRAINED_MODEL_ARCHIVE_MAP, CONFIG_NAME, BertConfig, BertLayerNorm, \
  WEIGHTS_NAME, TF_WEIGHTS_NAME, load_tf_weights_in_bert, BertEmbeddings, BertPooler, BertLayer
from relogic.utils.file_utils import cached_path
import torch
import shutil
import torch.nn as nn
import logging
import os
import tempfile
import tarfile
import copy


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BranchingBertPreTrainedModel(nn.Module):
  # def __init__(self, config):
  #   super(BranchingEncoder, self).__init__()
  #   self._config = config
  #   self.bert = BertModel(config)
  #   self.dropout = nn.Dropout(config.hidden_dropout_prob)
  #   self.apply(self.init_bert_weights)
  #
  # def forward(self, input_ids, token_type_ids=None, attention_mask=None,
  #             output_all_encoded_layers=False,
  #             output_final_multi_head_repr=False):
  #   sequence_output, _, final_multi_head_repr = self.bert(
  #     input_ids, token_type_ids, attention_mask,
  #     output_all_encoded_layers=output_all_encoded_layers,
  #     output_final_multi_head_repr=output_final_multi_head_repr)
  #
  #
  #
  #   if output_all_encoded_layers:
  #     sequence_output = [self.dropout(seq) for seq in sequence_output]
  #   else:
  #     sequence_output = self.dropout(sequence_output)
  #   if output_final_multi_head_repr:
  #     final_multi_head_repr = self.dropout(final_multi_head_repr)
  #     return sequence_output, final_multi_head_repr
  #   else:
  #     return sequence_output

  def __init__(self, config, *inputs, **kwargs):
    super(BranchingBertPreTrainedModel, self).__init__()
    if not isinstance(config, BertConfig):
      raise ValueError(
        "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
        "To create a model from a Google pretrained model use "
        "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
          self.__class__.__name__, self.__class__.__name__
        ))
    self.config = config

  def init_bert_weights(self, module):
    """ Initialize the weights.
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
      # Slightly different from the TF version which uses truncated_normal for initialization
      # cf https://github.com/pytorch/pytorch/pull/5617
      module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    elif isinstance(module, BertLayerNorm):
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
      module.bias.data.zero_()

  @classmethod
  def from_pretrained(cls, pretrained_model_name_or_path, state_dict=None, cache_dir=None,
                      from_tf=False, *inputs, **kwargs):
    if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
      archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
    else:
      archive_file = pretrained_model_name_or_path
    try:
      resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
    except EnvironmentError:
      logger.error(
        "Model name '{}' was not found in model name list ({}). "
        "We assumed '{}' was a path or url but couldn't find any file "
        "associated to this path or url.".format(
          pretrained_model_name_or_path,
          ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
          archive_file))
      return None
    if resolved_archive_file == archive_file:
      logger.info("loading archive file {}".format(archive_file))
    else:
      logger.info("loading archive file {} from cache at {}".format(
        archive_file, resolved_archive_file))

    tempdir = None
    if os.path.isdir(resolved_archive_file) or from_tf:
      serialization_dir = resolved_archive_file
    else:
      # Extract archive to temp dir
      tempdir = tempfile.mkdtemp()
      logger.info("extracting archive file {} to temp dir {}".format(
        resolved_archive_file, tempdir))
      with tarfile.open(resolved_archive_file, 'r:gz') as archive:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(archive, tempdir)
      serialization_dir = tempdir
    # Load config
    config_file = os.path.join(serialization_dir, CONFIG_NAME)
    config = BertConfig.from_json_file(config_file)
    logger.info("Model config {}".format(config))
    # Instantiate model.
    model = cls(config, *inputs, **kwargs)

    if state_dict is None and not from_tf:
      logger.info("Load model from torch model")
      weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
      state_dict = torch.load(weights_path, map_location='cpu' if not torch.cuda.is_available() else None)
    if tempdir:
      # Clean up temp dir
      shutil.rmtree(tempdir)
    if from_tf:
      logger.info("Load model from tensorflow model")
      # Directly load from a TensorFlow checkpoint
      weights_path = os.path.join(serialization_dir, TF_WEIGHTS_NAME)
      return load_tf_weights_in_bert(model, weights_path)
    # Load from a PyTorch state_dict
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
      new_key = None
      if 'gamma' in key:
        new_key = key.replace('gamma', 'weight')
      if 'beta' in key:
        new_key = key.replace('beta', 'bias')
      if new_key:
        old_keys.append(key)
        new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
      state_dict[new_key] = state_dict.pop(old_key)

    # Replicate the parameter for layer
    keys = list(state_dict.keys())
    for idx, width in enumerate(kwargs["encoder_structure"]):
      prefix = "bert.encoder.layer.{}.".format(idx)
      for key in keys:
        if key.startswith(prefix):
          for i in range(width):
            new_prefix = "bert.encoder.layer.{}.{}.".format(idx, i)
            new_key = key.replace(prefix, new_prefix)
            state_dict[new_key] = copy.deepcopy(state_dict[key])
          state_dict.pop(key)

    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
      state_dict._metadata = metadata

    def load(module, prefix=''):
      local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
      module._load_from_state_dict(
        state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
      for name, child in module._modules.items():
        if child is not None:
          load(child, prefix + name + '.')

    start_prefix = ''
    if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
      start_prefix = 'bert.'
    load(model, prefix=start_prefix)
    if len(missing_keys) > 0:
      logger.info("Weights of {} not initialized from pretrained model: {}".format(
        model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
      logger.info("Weights from pretrained model not used in {}: {}".format(
        model.__class__.__name__, unexpected_keys))
    if len(error_msgs) > 0:
      raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
        model.__class__.__name__, "\n\t".join(error_msgs)))
    return model


class BranchingBertEncoder(nn.Module):
  def __init__(self, config, encoder_structure):
    super(BranchingBertEncoder, self).__init__()
    layer = BertLayer(config)
    assert len(encoder_structure) == config.num_hidden_layers
    self.layer = nn.ModuleList([
      nn.ModuleList([copy.deepcopy(layer) for _ in range(encoder_structure[i])])
      for i in range(config.num_hidden_layers)])
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

  def forward(self,
              hidden_states,
              attention_mask,
              output_all_encoded_layers=True,
              output_final_multi_head_repr=False,
              route_path=None):
    all_encoder_layers = []
    for selected_route, layers in zip(route_path, self.layer):
      hidden_states, multi_head_repr = layers[selected_route](hidden_states, attention_mask)
      if output_all_encoded_layers:
        all_encoder_layers.append(hidden_states)
    if not output_all_encoded_layers:
      all_encoder_layers.append(hidden_states)
    if not output_final_multi_head_repr:
      return all_encoder_layers
    else:
      return all_encoder_layers, multi_head_repr


class BranchingBertModel(BranchingBertPreTrainedModel):
  def __init__(self, config, encoder_structure=None):
    super(BranchingBertModel, self).__init__(config)
    self.embeddings = BertEmbeddings(config)
    self.encoder = BranchingBertEncoder(config, encoder_structure)
    self.pooler = BertPooler(config)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.apply(self.init_bert_weights)

  def forward(self, input_ids, token_type_ids=None, attention_mask=None,
              output_all_encoded_layers=True, output_final_multi_head_repr=False,
              route_path=None):
    if attention_mask is None:
      attention_mask = torch.ones_like(input_ids)
    if token_type_ids is None:
      token_type_ids = torch.zeros_like(input_ids)

    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    # Share the embedding layer
    embedding_output = self.embeddings(input_ids, token_type_ids)
    encoded_layers, final_multi_head_repr = self.encoder(embedding_output,
                                    extended_attention_mask,
                                    output_all_encoded_layers=output_all_encoded_layers,
                                    output_final_multi_head_repr=True,
                                    route_path=route_path)
    pooled_output = self.pooler(encoded_layers[-1])
    if output_all_encoded_layers:
      sequence_output = [self.dropout(seq) for seq in encoded_layers]
    else:
      sequence_output = self.dropout(encoded_layers[-1])
    if not output_final_multi_head_repr:
      return sequence_output, None
    else:
      return sequence_output, final_multi_head_repr

if __name__ == "__main__":
  model = BranchingBertModel.from_pretrained(
    "bert-base-cased",
    encoder_structure=[3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3])
  # layer1 = model.encoder.layer[0]
  # print(layer1[0].attention.self.value.weight.data - layer1[1].attention.self.value.weight.data)
  feature, _, _ = model(torch.zeros(1, 5, dtype=torch.long),
                 output_all_encoded_layers=False,
                 route_path=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2])
  print(feature)

