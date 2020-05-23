from relogic.logickit.tasks.task import Task
from relogic.logickit.dataset.labeled_data_loader import LabeledDataLoader
from relogic.logickit.scorer.semparse.text_to_sql_scorer import (
  ColumnSelectionScorer, RATScorer, EditNetScorer, SlotFillingScorer, SQLRerankingScorer)
from relogic.logickit.base.constants import SLOT_FILLING_SQL_TASK, SQL_RERANKING_TASK, ZH_JOINT_CT_SQL_TASK

class TextToSQL(Task):
  def __init__(self, config, name, tokenizer=None):
    super().__init__(config, name, LabeledDataLoader(config, name, tokenizer))
    self.n_classes = len(self.loader.label_mapping.values())

  def get_module(self):
    pass

  def get_scorer(self, dump_to_file=None):
    if self.name in ["joint_ct_rat_sql", "rat_sql", ZH_JOINT_CT_SQL_TASK]:
      return RATScorer(dump_to_file=dump_to_file, dataflow=self.loader.dataflow)
    if self.name == "editnet_sql":
      return EditNetScorer(dump_to_file=dump_to_file, dataflow=self.loader.dataflow)
    if self.name == SLOT_FILLING_SQL_TASK:
      return SlotFillingScorer(dump_to_file=dump_to_file, dataflow=self.loader.dataflow)
    if self.name == SQL_RERANKING_TASK:
      return SQLRerankingScorer(dump_to_file=dump_to_file, dataflow=self.loader.dataflow)

    return ColumnSelectionScorer(dump_to_file=dump_to_file, dataflow=self.loader.dataflow)
