from relogic.logickit.tasks.task import Task
from relogic.logickit.dataset.labeled_data_loader import LabeledDataLoader
from relogic.logickit.modules.matching_module import MatchingModule
from relogic.logickit.modules.classification_module import ClassificationModule
from relogic.logickit.modules.representation_module import RepresentationModule
from relogic.logickit.modules.rel_extraction_module import RelExtractionModule
from relogic.logickit.modules.agg_matching_module import AggMatchingModule
from relogic.logickit.modules.ir_matching import IRMatchingModule
from relogic.logickit.scorer.ranking_scorer import RecallScorer, CartesianMatchingRecallScorer, RetrievalScorer
from relogic.logickit.scorer.classification_scorers import RelationF1Scorer, MultiClassAccuracyScorer
from relogic.logickit.modules.sam_cnn import SamCNN
from relogic.logickit.base.constants import IR_TASK, PAIRWISE_TASK, SINGLETON, ENTITY_TYPE_CLASSIFICATION, DOCIR_TASK, IR_SAMCNN_TASK


class Classification(Task):
  def __init__(self, config, name, tokenizer=None):
    super(Classification, self).__init__(
      config, name, LabeledDataLoader(config, name, tokenizer))
    self.n_classes = len(set(self.loader.label_mapping.values()))

  def get_module(self):
    if self.name in ["matching"]:
      return MatchingModule(self.config, self.name, self.n_classes)
    elif self.name in [IR_SAMCNN_TASK]:
      return SamCNN(self.config, self.name, self.n_classes)
    elif self.name.startswith(IR_TASK):
      if self.config.word_level_interaction:
        return IRMatchingModule(self.config, self.name, self.n_classes)
      else:
        return MatchingModule(self.config, self.name, self.n_classes)
    elif self.name.startswith(DOCIR_TASK):
      return AggMatchingModule(self.config, self.name, self.n_classes)
    elif self.name in ["pair_matching", PAIRWISE_TASK]:
      return RepresentationModule(self.config, self.name, self.config.repr_size)
    elif self.name in ["rel_extraction", SINGLETON]:
      return RelExtractionModule(self.config, self.name, self.n_classes)
    elif self.name in [ENTITY_TYPE_CLASSIFICATION]:
      return ClassificationModule(self.config, self.name, self.n_classes)


  def get_scorer(self, dump_to_file=None):
    if self.name in ["matching"]:
      return RecallScorer(self.loader.label_mapping, topk=self.config.topk, dump_to_file=dump_to_file)
    elif self.name.startswith(IR_TASK) or self.name.startswith(DOCIR_TASK):
      return RetrievalScorer(
        self.loader.label_mapping,
        qrels_file_path=self.config.qrels_file_path if isinstance(self.config.qrels_file_path, str)
            else self.config.tasks[self.name]["qrels_file_path"],
        dump_to_file=dump_to_file,
        regression=self.config.regression)
    elif self.name in ["rel_extraction"]:
      return RelationF1Scorer(self.loader.label_mapping, dump_to_file=dump_to_file)
    elif self.name in ["pair_matching", PAIRWISE_TASK]:
      return CartesianMatchingRecallScorer(
        topk=self.config.topk,
        qrels_file_path=self.config.qrels_file_path if isinstance(self.config.qrels_file_path, str)
          else self.config.tasks[self.name]["qrels_file_path"],
        dump_to_file=dump_to_file)
    elif self.name in [ENTITY_TYPE_CLASSIFICATION]:
      return MultiClassAccuracyScorer(self.loader.label_mapping, dump_to_file=dump_to_file, dataflow=self.loader.dataflow)
