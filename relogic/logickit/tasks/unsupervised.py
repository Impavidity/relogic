from relogic.logickit.tasks.task import Task
from relogic.logickit.dataset.labeled_data_loader import LabeledDataLoader
from relogic.logickit.modules.parallel_mapping_module import ParallelMappingModule
from relogic.logickit.modules.select_index_module import SelectIndexModule
from relogic.logickit.modules.mixsent_alignment import MixSentAlignmentModule
from relogic.logickit.modules.gen_repr import GenRepr
from relogic.logickit.base.constants import (PARALLEL_MAPPING_TASK,
  PARALLEL_TEACHER_STUDENT_TASK, MIXSENT_TASK, LANGUAGE_IDENTIFICATION_IR)
from relogic.logickit.scorer.distance_scorer import DistanceScorer



class Unsupervised(Task):
  def __init__(self, config, name, tokenizer=None):
    super(Unsupervised, self).__init__(
      config, name, LabeledDataLoader(config, name, tokenizer))
    self.config = config

  def get_module(self):
    if self.name in [PARALLEL_MAPPING_TASK]:
      return ParallelMappingModule(self.config, self.name)
    if self.name in [PARALLEL_TEACHER_STUDENT_TASK]:
      return SelectIndexModule(self.config, self.name)
    if self.name in [MIXSENT_TASK]:
      return MixSentAlignmentModule(self.config, self.name)
    if self.name in [LANGUAGE_IDENTIFICATION_IR]:
      return GenRepr(self.config, self.name)
    else:
      raise ValueError("Can not find task name {}".format(self.name))

  def get_scorer(self, dump_to_file=None):
    if self.name in [LANGUAGE_IDENTIFICATION_IR]:
      return None
    return DistanceScorer()