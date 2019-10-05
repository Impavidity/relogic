from relogic.logickit.tasks.task import Task
from relogic.logickit.dataset.labeled_data_loader import LabeledDataLoader
from relogic.logickit.modules.parallel_mapping_module import ParallelMappingModule
from relogic.logickit.base.constants import PARALLEL_MAPPING_TASK
from relogic.logickit.scorer.distance_scorer import DistanceScorer


class Unsupervised(Task):
  def __init__(self, config, name, tokenizer=None):
    super(Unsupervised, self).__init__(
      config, name, LabeledDataLoader(config, name, tokenizer))
    self.config = config

  def get_module(self):
    if self.name in [PARALLEL_MAPPING_TASK]:
      return ParallelMappingModule(self.config, self.name)
    else:
      raise ValueError("Can not find task name {}".format(self.name))

  def get_scorer(self, dump_to_file=None):
    return DistanceScorer()