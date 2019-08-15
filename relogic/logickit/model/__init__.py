from relogic.logickit.model.multitask_model import Model
from relogic.logickit.model.pair_matching import PairMatchingModel
# from model.branching_encoder import BranchingBertModel


def get_model(config):
  if "pair_matching" in config.task_names:
    return PairMatchingModel
  if "span_gcn_srl" in config.task_names:
    return
  return Model

