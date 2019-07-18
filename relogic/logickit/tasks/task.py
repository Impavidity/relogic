import abc


class Task(object, metaclass=abc.ABCMeta):
  def __init__(self, config, name, loader):
    self.config = config
    self.name = name
    self.loader = loader
    if config.mode == 'train' or config.mode == 'finetune':
      self.train_set = self.loader.get_dataset("train")
    else:
      self.train_set = None
    self.val_set = self.loader.get_dataset("dev" if (
          config.mode == 'train' or config.mode == 'valid' or config.mode == "finetune") else "test")

  @abc.abstractmethod
  def get_module(self):
    pass

  @abc.abstractmethod
  def get_scorer(self, dump_to_file=None):
    pass

