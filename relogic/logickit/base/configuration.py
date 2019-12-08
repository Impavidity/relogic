from dataclasses import dataclass
import json
from typing import Dict
import logging
from types import SimpleNamespace
import os
import relogic

logger = logging.getLogger(__name__)

class Configs:
  def __init__(self):
    pass

class EncoderConfigs(Configs):
  def __init__(self, encoder_configs: Dict):
    super().__init__()
    self.fix_embedding = encoder_configs.get("fix_embedding", False)

    self.fix_layers = []
    fix_layers = encoder_configs.get("fix_layers", list([]))
    for layer_range in fix_layers:
      if '-' in layer_range:
        s, e = layer_range.split('-')
        for i in range(int(s), int(e)+1): # inclusive
          self.fix_layers.append(i)
      else:
        self.fix_layers.append(int(layer_range))

class AdversarialConfigs(Configs):
  def __init__(self, adversarial_configs: Dict):
    super().__init__()
    self.activate = len(adversarial_configs) > 0
    self.type = adversarial_configs.get("type")
    self.discriminator_type = adversarial_configs.get("discriminator_type")
    self.discriminator_lr = adversarial_configs.get("adversarial_lr")
    self.soft_label = adversarial_configs.get("soft_label", False)
    self.clip_upper = adversarial_configs.get("clip_upper", 0.01)
    self.clip_lower = adversarial_configs.get("clip_lower", -0.01)
    self.scale = adversarial_configs.get("scale", 0.01)
    self.hidden_size = adversarial_configs.get("hidden_size", 798)




class Configuration:
  def __init__(
        self,
        tokenizer_configs: Dict,
        module_configs: Dict,
        task_configs: Dict,
        encoder_configs: EncoderConfigs,
        adversarial_configs: AdversarialConfigs):
    self.tokenizer_configs = tokenizer_configs
    self.module_configs = module_configs
    self.task_configs = task_configs
    self.encoder_configs = encoder_configs
    self.adversarial_configs = adversarial_configs
    if self.adversarial_configs.activate:
      logger.info(self.adversarial_configs.__dict__)


  @property
  def module_names(self):
    return self.module_configs.keys()

  @property
  def task_names(self):
    return self.task_configs.keys()

  @classmethod
  def load_from_json(cls, config):
    return cls(
      tokenizer_configs = config["tokenizers"],
      module_configs = config.get("modules"),
      task_configs = config.get("tasks"),
      encoder_configs = EncoderConfigs(config.get("encoder", dict({}))),
      adversarial_configs = AdversarialConfigs(config.get("adversarial", dict({})))
    )

  @classmethod
  def load_from_namespace(cls, config):
    pass

  @classmethod
  def load_from_json_file(cls, config_path):
    return cls.load_from_json(json.load(open(config_path)))

class Argument:
  def __init__(self):
    pass

  @classmethod
  def restore_from_model_path(cls, model_path, config_name="general_config.json", mode="deployment"):
    with open(os.path.join(model_path, config_name)) as f:
      restore_config = SimpleNamespace(**json.load(f))
    restore_config.mode = "deployment"
    restore_config.adversarial_training = False
    for task in restore_config.tasks:
      origin_label_path = restore_config.tasks[task]["label_mapping_path"]
      if origin_label_path != "none":
        label_mapping_path = origin_label_path.split("/")[-2:]
        absolute_label_mapping_path = os.path.join(os.path.dirname(relogic.__file__), "..", "data", *label_mapping_path)
        restore_config.tasks[task]["label_mapping_path"] = absolute_label_mapping_path
    restore_config.config_file = os.path.join(os.path.dirname(relogic.__file__), "..", restore_config.config_file)
    restore_config.training_scheme_file = os.path.join(os.path.dirname(relogic.__file__), "..", restore_config.training_scheme_file)
    return restore_config