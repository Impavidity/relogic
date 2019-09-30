import torch.nn as nn
import torch
from relogic.logickit.modules.span_extractors import EndpointSpanExtractor, AttentiveSpanExtractor 


class ECPExtractionModule(nn.Module):
  """Emotion-Cause Extraction
  """
  def __init__(self, config, task_name, n_classes):
    super(ECPExtractionModule, self).__init__()
    self.config = config
    self.task_name = task_name
    self.n_classes = n_classes

    self.endpoint_span_extractor = EndpointSpanExtractor(
      input_dim=config.hidden_size,
      num_width_embeddings=config.num_width_embeddings,
      span_width_embedding_dim=config.span_width_embedding_dim)
    
    hidden_size = config.hidden_size

    self.attentive_span_extractor = AttentiveSpanExtractor(input_dim=hidden_size)
  
  def forward(self, *inputs, **kwargs):
    features = kwargs.pop("features")
    clause_candidates = kwargs.pop("clause_candidates")

    clause_mask = (clause_candidates[:, :, 1] > 0).float()

    endpoint_span_embeddings = self.endpoint_span_extractor(
      sequence_tensor=features, span_indices=clause_candidates, span_indices_mask=clause_mask)

    raise NotImplementedError("Play with it")
    

    
    
