from relogic.logickit.tasks.tagging import Tagging
from relogic.logickit.tasks.classification import Classification
from relogic.logickit.tasks.span_extraction import SpanExtraction, SpanGCN
from relogic.logickit.tasks.span_extraction import ECPExtraction
from relogic.logickit.tasks.unsupervised import Unsupervised
from relogic.logickit.base.constants import (ECP_TASK, IR_TASK, NER_TASK, PARALLEL_MAPPING_TASK,
                                             PARALLEL_TEACHER_STUDENT_TASK, PAIRWISE_TASK)

def get_task(config, name, tokenizer):
  if name in ["ccg", "pos"]:
    return Tagging(config, name, True, tokenizer)
  elif name in ["chunk", "ner",
                "er", "nestedner", "srl",
                "srl_conll05", "srl_conll09",
                "srl_conll12", "predicate_sense", "joint_srl", NER_TASK]:
    return Tagging(config, name, False, tokenizer)
  elif name in ["matching", "rel_extraction", "pair_matching", IR_TASK, PAIRWISE_TASK]:
    return Classification(config, name, tokenizer)
  elif name in ["squad11", "squad20"]:
    return SpanExtraction(config, name, tokenizer)
  elif name in ["span_gcn"]:
    return SpanGCN(config, name, tokenizer)
  elif name in [ECP_TASK]:
    return ECPExtraction(config, name, tokenizer)
  elif name in [PARALLEL_MAPPING_TASK, PARALLEL_TEACHER_STUDENT_TASK]:
    return Unsupervised(config, name, tokenizer)
  elif name == "depparse":
    return DependencyParsing(config, name)
  else:
    raise ValueError("Unknow task", name)
