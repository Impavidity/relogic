from relogic.logickit.tasks.task import Task
from relogic.logickit.modules.srl_module import SRLModule
from relogic.logickit.modules.tagging_module import TaggingModule
from relogic.logickit.modules.predicate_sense_module import PredicateSenseModule
from relogic.logickit.scorer.tagging_scorers import EntityLevelF1Scorer, AccuracyScorer
from relogic.logickit.scorer.srl_scorers import SRLF1Scorer, SpanSRLF1Scorer
from relogic.logickit.dataset.labeled_data_loader import LabeledDataLoader

class Tagging(Task):
  def __init__(self, config, name, is_token_level=True, tokenizer=None):
    super(Tagging, self).__init__(
      config, name, LabeledDataLoader(config, name, tokenizer))
    self.n_classes = len(set(self.loader.label_mapping.values()))
    self.is_token_level = is_token_level

  def get_module(self):
    if self.name in ['srl', "srl_conll05", "srl_conll09", "srl_conll12"]:
      # if not self.config.predicate_surface_aware:
      #   return
      return SRLModule(self.config, self.name, self.n_classes)
    elif self.name in ['predicate_sense']:
      return PredicateSenseModule(self.config, self.name, self.n_classes)
    else:
      return TaggingModule(self.config, self.name, self.n_classes)

  def get_scorer(self, dump_to_file=None):
    if self.name in ["er", "ner"]:
      return EntityLevelF1Scorer(label_mapping=self.loader.label_mapping, dump_to_file=dump_to_file)
    if self.name in ["srl", "srl_conll05", "srl_conll09", "srl_conll12"]:
      if self.config.span_inference:
        return SpanSRLF1Scorer(label_mapping=self.loader.label_mapping, dump_to_file=dump_to_file)
      else:
        return SRLF1Scorer(label_mapping=self.loader.label_mapping, dump_to_file=dump_to_file)
    if self.name in ["predicate_sense"]:
      return AccuracyScorer(ignore_list=['O'], label_mapping=self.loader.label_mapping, dump_to_file=dump_to_file)