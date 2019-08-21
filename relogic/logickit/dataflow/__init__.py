from relogic.logickit.base.constants import SRL_TASK
from relogic.logickit.dataflow.dataflow import DataFlow, Example, Feature, MiniBatch
from relogic.logickit.dataflow.srl import SRLDataFlow, SRLExample, SRLFeature, SRLMiniBatch

TASK_TO_DATAFLOW_CLASS_MAP = {
  SRL_TASK: SRLDataFlow
}