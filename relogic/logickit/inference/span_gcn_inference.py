import torch.nn as nn

from relogic.logickit.base import utils
from relogic.logickit.inference.encoder import Encoder
from relogic.logickit.inference.branching_encoder import BranchingBertModel

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
    self.task_dict = dict([(task.name, task) for task in self.tasks])

  # def forward(self,
  #             task_name,
  #             input_ids,
  #             input_mask,
  #             input_head,
  #             segment_ids,
  #             label_ids,
  #             extra_args):
  def forward(self, *inputs, **kwargs):
    task_name = kwargs.pop("task_name")
    input_ids = kwargs.pop("input_ids")
    input_mask = kwargs.pop("input_mask")
    input_head = kwargs.pop("input_head")
    segment_ids = kwargs.pop("segment_ids")
    label_ids = kwargs.pop("label_ids")
    extra_args = kwargs.pop("extra_args", {})
    output_all_encoded_layers = extra_args.get("output_all_encoded_layers", False)
    route_path = extra_args.get("route_path", None)
    selected_non_final_layers = extra_args.get("selected_non_final_layers", None)
    no_dropout = task_name in READING_COMPREHENSION_TASKS

    features = self.encoder(
      input_ids=input_ids,
      token_type_ids=segment_ids,
      attention_mask=input_mask,
      output_all_encoded_layers=output_all_encoded_layers,
      selected_non_final_layers=selected_non_final_layers,
      route_path=route_path,
      no_dropout=no_dropout)

    # Semi-supervised Learning is not supported for now
    # There is a lot of semi-supervised learning algorithms,
    #  which are needed to be explored first, to make this
    #  framework more general

    # For each task, the interface is static
    #  including input_features, input_mask, segment_ids and extra_args
    logits = self.tasks_modules[task_name](
      features = features,
      input_mask=input_mask,
      segment_ids=segment_ids,
      extra_args=extra_args,
      **kwargs)

    if label_ids is not None:
      # if task_name in ["joint_srl"]:
      #   loss = self.task_dict[task_name].compute_loss()
      # else:
      loss = get_loss(
        task_name=task_name,
        logits=logits,
        label_ids=label_ids,
        config=self.config,
        extra_args=extra_args,
        **kwargs)
      return loss
    else:
      return logits






