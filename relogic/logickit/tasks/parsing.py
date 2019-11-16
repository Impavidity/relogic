from relogic.logickit.tasks.task import Task
from relogic.logickit.dataset.labeled_data_loader import LabeledDataLoader
from relogic.logickit.base.constants import DEP_PARSING_TASK
from relogic.logickit.modules.biaffine_dep_module import BiaffineDepModule
from relogic.logickit.scorer.dep_parsing_scorer import DepParsingScorer

class Parsing(Task):
  def __init__(self, config, name, tokenizer):
    super().__init__(
      config, name, LabeledDataLoader(config, name, tokenizer))
    self.n_classes = len(set(self.loader.label_mapping.values()))

  def get_module(self):
    if self.name in [DEP_PARSING_TASK]:
      return BiaffineDepModule(self.config, self.name, self.n_classes)
    else:
      raise NotImplementedError("The task {} is not defined".format(self.name))

  def get_scorer(self, dump_to_file=None):
    if self.name in [DEP_PARSING_TASK]:
      return DepParsingScorer(label_mapping=self.loader.label_mapping, dump_to_file=dump_to_file)