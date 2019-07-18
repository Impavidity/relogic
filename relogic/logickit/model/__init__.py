from relogic.logickit.model.encoder import Encoder
from relogic.logickit.model.multitask_model import Model
from relogic.logickit.model.pair_matching import PairMatchingModel
# from model.branching_encoder import BranchingBertModel

def get_encoder(config):
  if config.encoder_type == "normal":
    return Encoder.from_pretrained(config.bert_model)
  # if config.encoder_type == "branching":
  #   return BranchingBertModel(config)


def get_model(config):
  if "pair_matching" in config.task_names:
    return PairMatchingModel
  if "span_gcn_srl" in config.task_names:
    return
  return Model

# from inference.span_gcn_inference import SpanGCNInference