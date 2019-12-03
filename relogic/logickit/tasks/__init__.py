from relogic.logickit.tasks.tagging import Tagging
from relogic.logickit.tasks.classification import Classification
from relogic.logickit.tasks.span_extraction import SpanExtraction, SpanGCN
from relogic.logickit.tasks.span_extraction import ECPExtraction
from relogic.logickit.tasks.parsing import Parsing
from relogic.logickit.tasks.unsupervised import Unsupervised
from relogic.logickit.base.constants import (ECP_TASK, IR_TASK, NER_TASK, PARALLEL_MAPPING_TASK,
                                             PARALLEL_TEACHER_STUDENT_TASK, PAIRWISE_TASK, ENTITY_TYPE_CLASSIFICATION,
                                             DEP_PARSING_TASK, MIXSENT_TASK, LANGUAGE_IDENTIFICATION_IR, POS_TASK)
from relogic.common.prefix_map import PrefixMap

task_name_to_class_map = {
  POS_TASK: Tagging,
  NER_TASK: Tagging,
  IR_TASK: Classification,
  PAIRWISE_TASK: Classification,
  ENTITY_TYPE_CLASSIFICATION: Classification,
  ECP_TASK: ECPExtraction,
  PARALLEL_MAPPING_TASK: Unsupervised,
  PARALLEL_TEACHER_STUDENT_TASK: Unsupervised,
  MIXSENT_TASK: Unsupervised,
  DEP_PARSING_TASK:Parsing,
  LANGUAGE_IDENTIFICATION_IR: Unsupervised
}

TASK_NAME_TO_CLASS_MAP = PrefixMap(task_name_to_class_map)

def get_task(config, name, tokenizer):
  return TASK_NAME_TO_CLASS_MAP[name](config, name, tokenizer)

# def get_task(config, name, tokenizer):
#   if name in ["ccg", "pos"]:
#     return Tagging(config, name, True, tokenizer)
#   elif name in ["chunk", "ner",
#                 "er", "nestedner", "srl",
#                 "srl_conll05", "srl_conll09",
#                 "srl_conll12", "predicate_sense", "joint_srl", NER_TASK]:
#     return Tagging(config, name, False, tokenizer)
#   elif name in ["matching", "rel_extraction", "pair_matching", IR_TASK, PAIRWISE_TASK, ENTITY_TYPE_CLASSIFICATION]:
#     return Classification(config, name, tokenizer)
#   elif name in ["squad11", "squad20"]:
#     return SpanExtraction(config, name, tokenizer)
#   elif name in ["span_gcn"]:
#     return SpanGCN(config, name, tokenizer)
#   elif name in [ECP_TASK]:
#     return ECPExtraction(config, name, tokenizer)
#   elif name in [PARALLEL_MAPPING_TASK, PARALLEL_TEACHER_STUDENT_TASK, MIXSENT_TASK]:
#     return Unsupervised(config, name, tokenizer)
#   elif name in [DEP_PARSING_TASK]:
#     return Parsing(config, name, tokenizer)
#   else:
#     raise ValueError("Unknow task", name)


from relogic.logickit.dataset.labeled_data_loader import LabeledDataLoader
from relogic.logickit.dataflow import DataFlow

from relogic.logickit.base.constants import (
  NER_TASK
)
from relogic.logickit.modules.sequence_labeling_module import SequenceLabelingModule
from relogic.logickit.modules.matching_module import MatchingModule

# from relogic.logickit.scorer.ranking_scorer import RetrievalScorer
# import json
#
#
# task_name_to_scorer = {
#   IR_TASK: RetrievalScorer,# ,(self.loader.label_mapping, qrels_file_path=self.config.qrels_file_path, dump_to_file=dump_to_file)
# }
#
# TASK_NAME_TO_SCORER = PrefixMap(task_name_to_scorer)
#
# task_name_to_loss_func = {}
#
# TASK_NAME_TO_LOSS_FUNC = PrefixMap(task_name_to_loss_func)
#
# class Task(object):
#   """Unified Task Class
#   """
#   def __init__(self, name, config, task_config, tokenizers=None, data_loader=None):
#     self.config = config
#     self.name = name
#     self.loader = data_loader if data_loader is not None else LabeledDataLoader(config, name, tokenizers)
#     if config.mode == 'train' or config.mode == 'finetune':
#       self.train_set = self.loader.get_dataset("train")
#     else:
#       self.train_set = None
#     if config.mode != "deployment":
#       self.val_set = self.loader.get_dataset("dev" if (
#             config.mode == 'train' or config.mode == 'valid' or config.mode == "finetune") else "test")
#     else:
#       self.val_set = None
#     try:
#       self.dataset: DataFlow = self.loader.get_dataflow()
#     except:
#       self.dataset = None
#
#
#     self.task_config = json.load(open(config.module_configs[name]))
#     self.module = module if module is not None else TASK_NAME_TO_MODULE[name](config=config, **self.task_config)
#     # self.loss_func = TASK_NAME_TO_LOSS_FUNC[name](config=config, **self.task_config)
#
#   @property
#   def get_module(self):
#     return self.module
#
#   @classmethod
#   def from_config(cls, task_name, config=None, tokenizers=None, data_loader=None, task_config=None):
#     return cls(name=task_name, config=config, task_config=task_config, tokenizers=tokenizers, data_loader=data_loader)
#
#
#   def get_scorer(self):
#     return TASK_NAME_TO_SCORER[self.name](self.task_config)


