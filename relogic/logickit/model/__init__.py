from relogic.logickit.model.multitask_model import Model
from relogic.logickit.model.semparsing_model import SemParsingModel
from relogic.logickit.base.constants import SLOT_FILLING_SQL_TASK, SQL_RERANKING_TASK, ZH_JOINT_CT_SQL_TASK

SemParsingTasks = ["text_to_sql", "rat_sql", "editnet_sql", "joint_ct_rat_sql",
                   SLOT_FILLING_SQL_TASK, SQL_RERANKING_TASK, ZH_JOINT_CT_SQL_TASK]
def get_model(config):
  if any([task_name in SemParsingTasks for task_name in config.task_names]):
    return SemParsingModel
  return Model

