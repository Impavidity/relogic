import torch.nn as nn

from relogic.logickit.base import utils
from relogic.logickit.inference.encoder import Encoder
from relogic.logickit.inference.branching_encoder import BranchingBertModel
from relogic.logickit.modules.contextualizers.highway_lstm import HighwayLSTM

from relogic.logickit.base.constants import *

from relogic.logickit.loss_func import get_loss
from torch.utils.checkpoint import checkpoint
import numpy as np
import torch
from functools import reduce

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
    self.dummy_input = torch.rand(1, 10, requires_grad=True)

    # self.encoder = HighwayLSTM(num_layers=3, input_size=300, hidden_size=200, layer_dropout=0.2)
    # self.word_embedding = nn.Embedding(self.config.external_vocab_size, self.config.external_vocab_embed_size)
    # self.word_embedding.weight.data.copy_(torch.from_numpy(np.load(config.external_embeddings)))
    # print("Loading embedding from {}".format(config.external_embeddings))

  def encoding(self, **kwargs):
    task_name = kwargs.get("task_name", None)
    input_ids = kwargs.get("input_ids")
    input_mask = kwargs.get("input_mask")
    input_head = kwargs.get("input_head", None)
    segment_ids = kwargs.get("segment_ids")
    label_ids = kwargs.get("label_ids", None)
    extra_args = kwargs.get("extra_args", {})
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

    return features

  def decoding(self, **kwargs):
    task_name = kwargs.get("task_name")
    logits = self.tasks_modules[task_name](**kwargs)
    return logits

  def get_arguments(self, prefix, kwargs):
    arguments = {}
    task_name = kwargs.pop("task_name", None)
    arguments["input_ids"] = kwargs.pop(prefix+"input_ids")
    arguments["input_mask"] = kwargs.pop(prefix+"input_mask")
    arguments["input_head"] = kwargs.pop(prefix+"input_head", None)
    arguments["segment_ids"] = kwargs.pop(prefix+"segment_ids")
    arguments["label_ids"] = kwargs.pop("label_ids", None)
    arguments["extra_args"] = kwargs.pop("extra_args", {})
    arguments["output_all_encoded_layers"] = arguments["extra_args"].get("output_all_encoded_layers", False)
    arguments["route_path"] = arguments["extra_args"].get("route_path", None)
    arguments["selected_non_final_layers"] = arguments["extra_args"].get("selected_non_final_layers", None)
    arguments["no_dropout"] = task_name in READING_COMPREHENSION_TASKS
    return arguments


  def forward(self, *inputs, **kwargs):
    task_name = kwargs.get("task_name")

    if task_name == PARALLEL_TEACHER_STUDENT_TASK:
      teacher_prediction = kwargs.pop("teacher_predictions", None)
      if teacher_prediction is None:
        # Run teacher Part
        results = {}
        arguments = self.get_arguments(prefix="a_", kwargs=kwargs)
        features = self.encoding(**arguments)
        for task in self.tasks:
          if task.name != task_name:
            result = self.decoding(**arguments, **kwargs, features=features, task_name=task.name)
            results[task.name] = result.detach()
        return results
      else:
        arguments = self.get_arguments(prefix="b_", kwargs=kwargs)
        features = self.encoding(**arguments)
        losses = {}
        for task in self.tasks:
          if task.name != task_name:
            result = self.decoding(**arguments, features=features, task_name=task.name)
            logits, target, mask = self.decoding(
              task_name=task_name,
              student_results=result,
              teacher_results=teacher_prediction[task.name],
              **arguments, **kwargs)
            loss = get_loss(
              task=self.task_dict[task_name],
              logits=logits,
              config=self.config,
              target=target,
              mask=mask,
              **arguments, **kwargs)
            losses[task.name] = loss
        return reduce(lambda x,y:x+y, losses.values()), None

    # elif task_name in SIAMESE:
    #   return self.siamese_forward(*inputs, **kwargs)
    else:
      arguments = self.get_arguments(prefix="", kwargs=kwargs)
      features = self.encoding(**arguments)
      logits = self.decoding(**arguments, **kwargs, features=features, task_name=task_name)

      if arguments["label_ids"] is not None:
        # if task_name in ["joint_srl"]:
        #   loss = self.task_dict[task_name].compute_loss()
        # else:
        loss = get_loss(
          task=self.task_dict[task_name],
          logits=logits,
          config=self.config,
          **arguments, **kwargs)
        return loss, logits
      else:
        return logits.detach()


    # input_ids = kwargs.pop("input_ids")
    # input_mask = kwargs.pop("input_mask")
    # input_head = kwargs.pop("input_head", None)
    # segment_ids = kwargs.pop("segment_ids")
    # label_ids = kwargs.pop("label_ids", None)
    # extra_args = kwargs.pop("extra_args", {})
    # output_all_encoded_layers = extra_args.get("output_all_encoded_layers", False)
    # route_path = extra_args.get("route_path", None)
    # selected_non_final_layers = extra_args.get("selected_non_final_layers", None)
    # no_dropout = task_name in READING_COMPREHENSION_TASKS

    # BERT encoding
    # features = self.encoder(
    #   input_ids=input_ids,
    #   token_type_ids=segment_ids,
    #   attention_mask=input_mask,
    #   output_all_encoded_layers=output_all_encoded_layers,
    #   selected_non_final_layers=selected_non_final_layers,
    #   route_path=route_path,
    #   no_dropout=no_dropout)

    # features = checkpoint(self.encoder, {
    #   "input_ids": input_ids,
    #   "token_type_ids": segment_ids,
    #   "attention_mask": input_mask,
    #   "output_all_encoded_layers": output_all_encoded_layers,
    #   "selected_non_final_layers": selected_non_final_layers,
    #   "route_path": route_path,
    #   "no_dropout": no_dropout,
    # })

    # # LSTM encoder
    # input_token_ids = self.word_embedding(kwargs.pop("_input_token_ids"))
    # token_lengths = kwargs.pop("_token_length")
    # label_ids = kwargs.pop("_label_ids")
    # features = self.encoder(input_token_ids, token_lengths)

    # Semi-supervised Learning is not supported for now
    # There is a lot of semi-supervised learning algorithms,
    #  which are needed to be explored first, to make this
    #  framework more general

    # For each task, the interface is static
    #  including input_features, input_mask, segment_ids and extra_args

    # logits = self.tasks_modules[task_name](
    #   features = features,
    #   input_mask=input_mask,
    #   segment_ids=segment_ids,
    #   extra_args=extra_args,
    #   **kwargs)

    # logits = checkpoint(self.tasks_modules[task_name], {
    #   "features": features,
    #   "input_mask": input_mask,
    #   "segment_ids": segment_ids,
    #   "extra_args": extra_args,
    # }.update(kwargs))

    # if label_ids is not None:
    #   # if task_name in ["joint_srl"]:
    #   #   loss = self.task_dict[task_name].compute_loss()
    #   # else:
    #   loss = get_loss(
    #     task=self.task_dict[task_name],
    #     logits=logits,
    #     label_ids=label_ids,
    #     input_head=input_head,
    #     config=self.config,
    #     extra_args=extra_args,
    #     **kwargs)
    #   return loss, logits
    # else:
    #   return logits.detach()

  def siamese_forward(self, *inputs, **kwargs):
    task_name = kwargs.pop("task_name")
    a_input_ids = kwargs.pop("a_input_ids")
    a_input_mask = kwargs.pop("a_input_mask")
    a_segment_ids = kwargs.pop("a_segment_ids")
    b_input_ids = kwargs.pop("b_input_ids")
    b_input_mask = kwargs.pop("b_input_mask")
    b_segment_ids = kwargs.pop("b_segment_ids")
    extra_args = kwargs.pop("extra_args", {})
    output_all_encoded_layers = extra_args.get("output_all_encoded_layers", False)
    route_path = extra_args.get("route_path", None)
    selected_non_final_layers = extra_args.get("selected_non_final_layers", None)

    # BERT encoding
    a_features = self.encoder(
      input_ids=a_input_ids,
      token_type_ids=a_segment_ids,
      attention_mask=a_input_mask,
      output_all_encoded_layers=output_all_encoded_layers,
      selected_non_final_layers=selected_non_final_layers,
      route_path=route_path)

    b_features = self.encoder(
      input_ids=b_input_ids,
      token_type_ids=b_segment_ids,
      attention_mask=b_input_mask,
      output_all_encoded_layers=output_all_encoded_layers,
      selected_non_final_layers=selected_non_final_layers,
      route_path=route_path)

    logits = self.tasks_modules[task_name](
      a_features=a_features,
      b_features=b_features,
      extra_args=extra_args,
      **kwargs)

    loss = (logits * logits).sum() / logits.size(0)

    return loss, logits






