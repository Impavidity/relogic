from relogic.logickit.tasks.task import Task
from relogic.logickit.dataset.labeled_data_loader import LabeledDataLoader
from relogic.logickit.modules.matching_module import MatchingModule
from relogic.logickit.modules.representation_module import RepresentationModule
from relogic.logickit.modules.rel_extraction_module import RelExtractionModule
from relogic.logickit.scorer.ranking_scorer import RecallScorer, CartesianMatchingRecallScorer
from relogic.logickit.scorer.classification_scorers import RelationF1Scorer


class Classification(Task):
  def __init__(self, config, name, tokenizer=None):
    super(Classification, self).__init__(
      config, name, LabeledDataLoader(config, name, tokenizer))
    self.n_classes = len(set(self.loader.label_mapping.values()))

  def get_module(self):
    if self.name in ["matching"]:
      return MatchingModule(self.config, self.name, self.n_classes)
    elif self.name in ["ir"]:
      pass
    elif self.name in ["pair_matching"]:
      return RepresentationModule(self.config, self.name, self.config.repr_size)
    elif self.name in ["rel_extraction"]:
      return RelExtractionModule(self.config, self.name, self.n_classes)


  def get_scorer(self, dump_to_file=None):
    if self.name in ["matching"]:
      return RecallScorer(self.loader.label_mapping, topk=self.config.topk, dump_to_file=dump_to_file)
    elif self.name in ["ir"]:
      return
    elif self.name in ["rel_extraction"]:
      return RelationF1Scorer(self.loader.label_mapping, dump_to_file=dump_to_file)
    elif self.name in ["pair_matching"]:
      return CartesianMatchingRecallScorer(topk=self.config.topk, dump_to_file=dump_to_file)