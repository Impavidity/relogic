from typing import List
from relogic.structures.structure import Structure
from relogic.utils.file_utils import cached_path, RELOGIC_CACHE
import logging
import os
import json
from types import SimpleNamespace
from relogic.logickit.training.predictor import Predictor
from relogic.logickit.base.configuration import Argument




logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
  "ner-zh": "",
  "ner-en": "",
  "entity-linking": "",
  "srl-conll12": "https://git.uwaterloo.ca/p8shi/data-server/raw/master/srl_conll12_mlp/default.ckpt"
}

PRETRAINED_CONFIG_ARCHIVE_MAP = {
  "ner-zh": "",
  "ner-en": "",
  "entity-linking": "",
  "srl-conll12": "https://git.uwaterloo.ca/p8shi/data-server/raw/master/srl_conll12_mlp/general_config.json"
}

WEIGHTS_NAME = "default.ckpt"
CONFIG_NAME = "general_config.json"

class Component(object):
  """

  """
  def __init__(self, config, predictor: Predictor=None):
    self.config = config
    self._predictor = predictor

  def execute(self, inputs: List[Structure]):
    raise NotImplementedError()

  @classmethod
  def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
    if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
      archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
      config_file = PRETRAINED_CONFIG_ARCHIVE_MAP[pretrained_model_name_or_path]
    else:
      archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
      config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)

    # redirect to the cache, if necessary
    try:
      resolved_archive_file = cached_path(archive_file, cache_dir=RELOGIC_CACHE)
    except EnvironmentError:
      if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
        logger.error(
          "Couldn't reach server at '{}' to download pretrained weights.".format(
            archive_file))
      else:
        logger.error(
          "Model name '{}' was not found in model name list ({}). "
          "We assumed '{}' was a path or url but couldn't find any file "
          "associated to this path or url.".format(
            pretrained_model_name_or_path,
            ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
            archive_file))
      return None

    try:
      resolved_config_file = cached_path(config_file, cache_dir=RELOGIC_CACHE)
    except EnvironmentError:
      if pretrained_model_name_or_path in PRETRAINED_CONFIG_ARCHIVE_MAP:
        logger.error(
          "Couldn't reach server at '{}' to download pretrained model configuration file.".format(
            config_file))
      else:
        logger.error(
          "Model name '{}' was not found in model name list ({}). "
          "We assumed '{}' was a path or url but couldn't find any file "
          "associated to this path or url.".format(
            pretrained_model_name_or_path,
            ', '.join(PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
            config_file))
      return None

    if resolved_archive_file == archive_file and resolved_config_file == config_file:
      logger.info("loading weights file {}".format(archive_file))
      logger.info("loading configuration file {}".format(config_file))
    else:
      logger.info("loading weights file {} from cache at {}".format(
        archive_file, resolved_archive_file))
      logger.info("loading configuration file {} from cache at {}".format(
        config_file, resolved_config_file))


    resolved_config_file_dir = os.path.dirname(resolved_config_file)
    resolved_config_file_name = os.path.basename(resolved_config_file)
    restore_config = Argument.restore_from_model_path(model_path=resolved_config_file_dir, config_name=resolved_config_file_name)
    predictor = Predictor(restore_config)
    resolved_model_file_dir = os.path.dirname(resolved_archive_file)
    resolved_model_file_name = os.path.basename(resolved_archive_file)
    predictor.restore(model_path=resolved_model_file_dir, model_name=resolved_model_file_name)
    return cls(config=restore_config, predictor=predictor)