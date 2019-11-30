import abc
from relogic.logickit.dataflow import DataFlow


class Task(object, metaclass=abc.ABCMeta):
  def __init__(self, config, name, loader):
    self.config = config
    self.name = name
    self.loader = loader
    self.has_module = True
    self.has_scorer = True
    if config.mode == 'train' or config.mode == 'finetune':
      self.train_set = self.loader.get_dataset("train")
    else:
      self.train_set = None
    if config.mode != "deployment":
      self.val_set = self.loader.get_dataset("dev" if (
            config.mode == 'train' or config.mode == 'valid' or config.mode == "finetune") else "test")
    else:
      self.val_set = None
    try:
      self.dataset: DataFlow = self.loader.get_dataflow()
    except:
      self.dataset = None

  @abc.abstractmethod
  def get_module(self):
    pass

  @abc.abstractmethod
  def get_scorer(self, dump_to_file=None):
    pass



