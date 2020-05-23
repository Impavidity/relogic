import os
from relogic.logickit.arguments.text_to_sql import TextToSQLArguments, RATSQLArguments, EditNetArguments, SQLRerankingArgument
from relogic.logickit.base.constants import SLOT_FILLING_SQL_TASK, SQL_RERANKING_TASK, ZH_JOINT_CT_SQL_TASK

TASK_NAME_TO_ARGUMENT = {
  "text_to_sql": TextToSQLArguments,
  "rat_sql": RATSQLArguments,
  "joint_ct_rat_sql": RATSQLArguments,
  "editnet_sql": EditNetArguments,
  SLOT_FILLING_SQL_TASK: RATSQLArguments,
  SQL_RERANKING_TASK: SQLRerankingArgument,
  ZH_JOINT_CT_SQL_TASK: RATSQLArguments,
}

def add_task_specific_args(task_names, parser):
  for task_name in task_names:
    if task_name in TASK_NAME_TO_ARGUMENT:
      TASK_NAME_TO_ARGUMENT[task_name].add_task_specific_args(parser, os.getcwd())
      TASK_NAME_TO_ARGUMENT[task_name].add_model_specific_args(parser, os.getcwd())

