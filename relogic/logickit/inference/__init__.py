from relogic.logickit.inference.encoder import Encoder
from relogic.logickit.inference.inference import Inference

def get_inference(config):
  # if config.span_inference:
  #   return SpanGCNInference
  # else:
  #   return Inference
  return Inference


def get_encoder(config):
  if config.encoder_type == "normal":
    return Encoder.from_pretrained(config.bert_model)
  # if config.encoder_type == "branching":
  #   return BranchingBertModel(config)