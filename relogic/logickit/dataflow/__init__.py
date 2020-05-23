from relogic.logickit.base.constants import (JOINT_SRL_TASK, PIPE_SRL_TASK, ECP_TASK, POINTWISE_TASK, IR_TASK, NER_TASK,
  SEQUENCE_LABELING_TASK, PARALLEL_MAPPING_TASK, PAIRWISE_TASK, PARALLEL_TEACHER_STUDENT_TASK,
  SEQUENCE_CLASSIFICATION_TASK, ENTITY_TYPE_CLASSIFICATION, DEP_PARSING_TASK, MIXSENT_TASK, LANGUAGE_IDENTIFICATION_IR,
  POS_TASK, DOCIR_TASK, LANGUAGE_IDENTIFICATION_SEQ, PREDICATE_DETECTION_TASK, IR_SIAMESE_TASK, IR_SAMCNN_TASK,
  TEXT_TO_SQL_TASK, RAT_SQL_TASK, EDITNET_SQL_TASK, JOINT_CT_RAT_SQL_TASK, SLOT_FILLING_SQL_TASK, SQL_RERANKING_TASK,
  ZH_JOINT_CT_SQL_TASK)
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
from relogic.logickit.dataflow.doc_pointwise import (DocPointwiseDataFlow,
  DocPointwiseExample, DocPointwiseFeature, DocPointwiseMiniBatch)
from relogic.logickit.dataflow.twins import TwinsDataFlow, TwinsExample, TwinsFeature, TwinsMiniBatch
from relogic.logickit.dataflow.semparse.column_selection import (ColumnSelectionDataFlow, ColumnSelectionExample,
                                                                 ColumnSelectionFeature, ColumnSelectionMiniBatch)
from relogic.logickit.dataflow.semparse.rat_sql import (RATSQLDataFlow, RATSQLExample, RATSQLFeature, RATSQLMiniBatch)
from relogic.logickit.dataflow.semparse.editnet import (EditNetDataFlow, EditNetExample, EditNetFeature, EditNetMiniBatch)
from relogic.logickit.dataflow.semparse.joint_ct import (JointCTRATSQLDataFlow, JointCTRATSQLExample, JointCTRATSQLFeature,
                                                         JointCTRATSQLMiniBatch)
from relogic.logickit.dataflow.semparse.slot_filling import (SlotFillingDataFlow, SlotFillingFeature, SlotFillingExample,
                                                             SlotFillingMiniBatch)
from relogic.logickit.dataflow.semparse.reranking import (RerankingDataFlow, RerankingFeature, RerankingExample, RerankingMiniBatch)
from relogic.logickit.dataflow.semparse.zh_joint_ct import (ZHJointCTSQLFeature, ZHJointCTSQLExample, ZHJointCTSQLMiniBatch,
                                                            ZHJointCTSQLDataFlow)
from relogic.common.prefix_map import PrefixMap

task_to_dataflow_class_map = {
  PIPE_SRL_TASK: SRLDataFlow,
  JOINT_SRL_TASK: SRLDataFlow,
  ECP_TASK: ECPDataFlow,
  POINTWISE_TASK: PointwiseDataFlow,
  IR_TASK: PointwiseDataFlow,
  IR_SIAMESE_TASK: TwinsDataFlow,
  IR_SAMCNN_TASK: TwinsDataFlow,
  SEQUENCE_LABELING_TASK: SequenceDataFlow,
  POS_TASK: SequenceDataFlow,
  NER_TASK: SequenceDataFlow,
  PREDICATE_DETECTION_TASK: SequenceDataFlow,
  PARALLEL_MAPPING_TASK: ParallelDataFlow,
  PAIRWISE_TASK: PairwiseDataFlow,
  PARALLEL_TEACHER_STUDENT_TASK: ParallelDataFlow,
  SEQUENCE_CLASSIFICATION_TASK: SingletonDataFlow,
  ENTITY_TYPE_CLASSIFICATION: SingletonDataFlow,
  DEP_PARSING_TASK: DependencyParsingDataFlow,
  MIXSENT_TASK: MixSentDataFlow,
  LANGUAGE_IDENTIFICATION_IR: PointwiseDataFlow,
  LANGUAGE_IDENTIFICATION_SEQ: SingletonDataFlow,
  DOCIR_TASK: DocPointwiseDataFlow,
  TEXT_TO_SQL_TASK: ColumnSelectionDataFlow,
  RAT_SQL_TASK: RATSQLDataFlow,
  JOINT_CT_RAT_SQL_TASK: JointCTRATSQLDataFlow,
  EDITNET_SQL_TASK: EditNetDataFlow,
  SLOT_FILLING_SQL_TASK: SlotFillingDataFlow,
  SQL_RERANKING_TASK: RerankingDataFlow,
  ZH_JOINT_CT_SQL_TASK: ZHJointCTSQLDataFlow
}

TASK_TO_DATAFLOW_CLASS_MAP = PrefixMap(task_to_dataflow_class_map)