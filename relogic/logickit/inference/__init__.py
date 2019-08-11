from relogic.logickit.inference.span_gcn_inference import SpanGCNInference
from relogic.logickit.model.inference import Inference

def get_inference(config):
  if config.span_inference:
    return SpanGCNInference
  else:
    return Inference