from relogic.logickit.base.constants import (SRL_TASK, ECP_TASK, POINTWISE_TASK, IR_TASK, NER_TASK,
  SEQUENCE_LABELING_TASK, PARALLEL_MAPPING_TASK, PAIRWISE_TASK, PARALLEL_TEACHER_STUDENT_TASK,
  SEQUENCE_CLASSIFICATION_TASK, ENTITY_TYPE_CLASSIFICATION, DEP_PARSING_TASK, MIXSENT_TASK, LANGUAGE_IDENTIFICATION_IR,
  POS_TASK)
from relogic.logickit.dataflow.dataflow import DataFlow, Example, Feature, MiniBatch
from relogic.logickit.dataflow.srl import SRLDataFlow, SRLExample, SRLFeature, SRLMiniBatch
from relogic.logickit.dataflow.ecp import ECPDataFlow, ECPExample, ECPFeature, ECPMiniBatch
from relogic.logickit.dataflow.pointwise import PointwiseDataFlow, PointwiseExample, PointwiseFeature, PointwiseMiniBatch
from relogic.logickit.dataflow.sequence import SequenceDataFlow, SequenceExample, SequenceFeature, SequenceMiniBatch
from relogic.logickit.dataflow.parallel import ParallelDataFlow, ParallelExample, ParallelFeature, ParallelMiniBatch
from relogic.logickit.dataflow.pairwise import PairwiseDataFlow, PairwiseExample, PairwiseFeature, PairwiseMiniBatch
from relogic.logickit.dataflow.singleton import SingletonDataFlow, SingletonExample, SingletonFeature, SingletonMiniBatch
from relogic.logickit.dataflow.dep import (DependencyParsingDataFlow,
  DependencyParsingExample, DependencyParsingFeature, DependencyParsingMiniBatch)
from relogic.logickit.dataflow.mixsent import MixSentDataFlow, MixSentExample, MixSentFeature, MixSentMiniBatch
from relogic.common.prefix_map import PrefixMap

task_to_dataflow_class_map = {
  SRL_TASK: SRLDataFlow,
  ECP_TASK: ECPDataFlow,
  POINTWISE_TASK: PointwiseDataFlow,
  IR_TASK: PointwiseDataFlow,
  SEQUENCE_LABELING_TASK: SequenceDataFlow,
  POS_TASK: SequenceDataFlow,
  NER_TASK: SequenceDataFlow,
  PARALLEL_MAPPING_TASK: ParallelDataFlow,
  PAIRWISE_TASK: PairwiseDataFlow,
  PARALLEL_TEACHER_STUDENT_TASK: ParallelDataFlow,
  SEQUENCE_CLASSIFICATION_TASK: SingletonDataFlow,
  ENTITY_TYPE_CLASSIFICATION: SingletonDataFlow,
  DEP_PARSING_TASK: DependencyParsingDataFlow,
  MIXSENT_TASK: MixSentDataFlow,
  LANGUAGE_IDENTIFICATION_IR: PointwiseDataFlow
}

TASK_TO_DATAFLOW_CLASS_MAP = PrefixMap(task_to_dataflow_class_map)