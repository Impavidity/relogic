from dataclasses import dataclass
import json
from typing import Dict

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
    print("----")
    print(self.adversarial_configs.__dict__)


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


