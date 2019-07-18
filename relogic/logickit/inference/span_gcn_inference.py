import torch
import torch.nn as nn

from relogic.logickit.base import utils
from relogic.logickit.model.encoder import Encoder
from relogic.logickit.model.branching_encoder import BranchingBertModel

from relogic.logickit.base.constants import *

from relogic.logickit.loss_func import get_loss

class SpanGCNInference(nn.Module):
  """
  The inference module is to
    1. Run the inference process
    2. Calculate the loss
    3. Return the logits or loss depends on the mode (train or test)
  """
  def __init__(self, config, tasks):
    super(SpanGCNInference, self).__init__()
    self.config = config
    self.tasks = tasks
    if config.branching_encoder:
      utils.log("Build Branching Bert Encoder")
      self.encoder = BranchingBertModel.from_pretrained(
        config.bert_model, encoder_structure=config.branching_structure)
    else:
      utils.log("Build Bert Encoder")
      self.encoder = Encoder.from_pretrained(config.bert_model)
    utils.log("Build Task Modules")
    self.tasks_modules = nn.ModuleDict()
    for task in tasks:
      self.tasks_modules.update([(task.name, task.get_module())])

  def forward(self,
              task_name,
              input_ids,
              input_mask,
              segment_ids,
              label_ids,
              extra_args):

    output_all_encoded_layers = extra_args.get("output_all_encoded_layers", False)
    output_final_multi_head_repr = extra_args.get("output_final_multi_head_repr", False)
    route_path = extra_args.get("route_path", None)
    selected_non_final_layers = extra_args.get("selected_non_final_layers", None)
    no_dropout = task_name in READING_COMPREHENSION_TASKS

    features = self.encoder(
      input_ids=input_ids,
      token_type_ids=segment_ids,
      attention_mask=input_mask,
      output_all_encoded_layers=output_all_encoded_layers,
      output_final_multi_head_repr=output_final_multi_head_repr,
      selected_non_final_layers=selected_non_final_layers,
      route_path=route_path,
      no_dropout=no_dropout)
    if output_final_multi_head_repr:
      sequence_output, final_multi_head_repr = features
    else:
      sequence_output = features
    if not output_all_encoded_layers and selected_non_final_layers is None:
      features = sequence_output
      selected_non_final_layers = None
    else:
      features = sequence_output[-1]
      selected_non_final_layers = sequence_output[:-1]
    extra_args["selected_non_final_layers"] = selected_non_final_layers

    # Semi-supervised Learning is not supported for now
    # There is a lot of semi-supervised learning algorithms,
    #  which are needed to be explored first, to make this
    #  framework more general

    # For each task, the interface is static
    #  including input_features, input_mask, segment_ids and extra_args
    logits = self.tasks_modules[task_name](
      features,
      input_mask=input_mask,
      segment_ids=segment_ids,
      extra_args=extra_args)

    if label_ids is not None:
      loss = get_loss(
        task_name=task_name,
        logits=logits,
        label_ids=label_ids,
        config=self.config)
      return loss
    else:
      return logits





