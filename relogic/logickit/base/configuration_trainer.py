from dataclasses import dataclass
import json
from typing import Dict

@dataclass
class TrainerConfig:

  tokenizers: Dict

  def __post_init__(self):
    pass

  @classmethod
  def load_from_json(cls, config):
    return cls(
      tokenizers = config["tokenizers"]
    )

  @classmethod
  def load_from_namespace(cls, config):
    pass

  @classmethod
  def load_from_json_file(cls, config_path):
    return cls.load_from_json(json.load(open(config_path)))


