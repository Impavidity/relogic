from relogic.logickit.base.constants import (SRL_TASK, ECP_TASK, POINTWISE_TASK, IR_TASK, NER_TASK,
  SEQUENCE_LABELING_TASK, PARALLEL_MAPPING_TASK, PARALLEL_TEACHER_STUDENT_TASK)
from relogic.logickit.dataflow.dataflow import DataFlow, Example, Feature, MiniBatch
from relogic.logickit.dataflow.srl import SRLDataFlow, SRLExample, SRLFeature, SRLMiniBatch
from relogic.logickit.dataflow.ecp import ECPDataFlow, ECPExample, ECPFeature, ECPMiniBatch
from relogic.logickit.dataflow.pointwise import PointwiseDataFlow, PointwiseExample, PointwiseFeature, PointwiseBatch
from relogic.logickit.dataflow.sequence import SequenceDataFlow, SequenceExample, SequenceFeature, SequenceMiniBatch
from relogic.logickit.dataflow.parallel import ParallelDataFlow, ParallelExample, ParallelFeature, ParallelMiniBatch

TASK_TO_DATAFLOW_CLASS_MAP = {
  SRL_TASK: SRLDataFlow,
  ECP_TASK: ECPDataFlow,
  POINTWISE_TASK: PointwiseDataFlow,
  IR_TASK: PointwiseDataFlow,
  SEQUENCE_LABELING_TASK: SequenceDataFlow,
  NER_TASK: SequenceDataFlow,
  PARALLEL_MAPPING_TASK: ParallelDataFlow,
  PARALLEL_TEACHER_STUDENT_TASK: ParallelDataFlow
}