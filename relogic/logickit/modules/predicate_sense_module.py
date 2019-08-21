import torch.nn as nn
import torch
class PredicateSenseModule(nn.Module):
  def __init__(self, config, task_name, n_classes):
    super(PredicateSenseModule, self).__init__()
    self.config = config
    self.predicate_indicator_embedding = nn.Embedding(2, 10)
    self.linear = nn.Linear(in_features=config.hidden_size + 10, out_features=n_classes)

  def forward(self,
              input,
              view_type="primary",
              final_multi_head_repr=None,
              input_mask=None,
              segment_ids=None,
              extra_args=None,
              **kwargs):
    # hard code
    predicate_indicator = extra_args["is_predicate_id"].long()
    predicate_ind_embed = self.predicate_indicator_embedding(predicate_indicator)
    logits = self.linear(torch.cat([input, predicate_ind_embed], dim=-1))
    return logits
