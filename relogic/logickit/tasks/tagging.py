from relogic.logickit.tasks.task import Task
from relogic.logickit.modules.srl_module import SRLModule
from relogic.logickit.modules.tagging_module import TaggingModule
from relogic.logickit.modules.predicate_sense_module import PredicateSenseModule
from relogic.logickit.modules.joint_srl_module import JointSRLModule
from relogic.logickit.modules.sequence_labeling_module import SequenceLabelingModule
from relogic.logickit.scorer.tagging_scorers import EntityLevelF1Scorer, AccuracyScorer
from relogic.logickit.scorer.srl_scorers import SRLF1Scorer, SpanSRLF1Scorer, JointSpanSRLF1Scorer
from relogic.logickit.dataset.labeled_data_loader import LabeledDataLoader
from relogic.logickit.base.constants import SEQUENCE_LABELING_TASK, NER_TASK, POS_TASK

class Tagging(Task):
  def __init__(self, config, name,  tokenizer=None):
    super(Tagging, self).__init__(
      config, name, LabeledDataLoader(config, name, tokenizer))
    self.n_classes = len(set(self.loader.label_mapping.values()))

  def get_module(self):
    if self.name in ['srl', "srl_conll05", "srl_conll09", "srl_conll12"]:
      # if not self.config.predicate_surface_aware:
      #   return
      return SRLModule(self.config, self.name, self.n_classes)
    if self.name in ["joint_srl"]:
      return JointSRLModule(self.config, self.name, self.n_classes)
    elif self.name in ['predicate_sense']:
      return PredicateSenseModule(self.config, self.name, self.n_classes)
    elif self.name in [NER_TASK, POS_TASK]:
      return SequenceLabelingModule(self.config, self.name, self.n_classes)
    else:
      return TaggingModule(self.config, self.name, self.n_classes)
      # raise ValueError("Task name {} is not defined".format(self.name))

  def get_scorer(self, dump_to_file=None):
    if self.name in ["er", "ner", NER_TASK]:
      return EntityLevelF1Scorer(label_mapping=self.loader.label_mapping, dump_to_file=dump_to_file)
    if self.name in ["srl", "srl_conll05", "srl_conll09", "srl_conll12"]:
      if self.config.span_inference:
        return SpanSRLF1Scorer(label_mapping=self.loader.label_mapping, dump_to_file=dump_to_file)
      else:
        return SRLF1Scorer(label_mapping=self.loader.label_mapping, dump_to_file=dump_to_file)
    if self.name in ["joint_srl"]:
      return JointSpanSRLF1Scorer(label_mapping=self.loader.label_mapping, dump_to_file=dump_to_file)
    if self.name in ["predicate_sense"]:
      return AccuracyScorer(ignore_list=['O'], label_mapping=self.loader.label_mapping, dump_to_file=dump_to_file)
    if self.name in [POS_TASK]:
      return AccuracyScorer(label_mapping=self.loader.label_mapping, dump_to_file=dump_to_file)