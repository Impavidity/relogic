import torch.nn as nn

from relogic.logickit.base import utils
from relogic.logickit.inference.encoder import get_encoder
from relogic.logickit.inference.branching_encoder import BranchingBertModel
from torch.nn import MarginRankingLoss
from relogic.logickit.modules.contextualizers.highway_lstm import HighwayLSTM

from relogic.logickit.base.constants import *

from relogic.logickit.loss_func import get_loss
from torch.utils.checkpoint import checkpoint
import numpy as np
import torch
from functools import reduce

class Inference(nn.Module):
  """
  The inference module is to
    1. Run the inference process
    2. Calculate the loss
    3. Return the logits or loss depends on the mode (train or test)
  """
  def __init__(self, config, tasks):
    super(Inference, self).__init__()
    self.config = config
    self.tasks = tasks
    if config.branching_encoder:
      utils.log("Build Branching Bert Encoder")
      self.encoder = BranchingBertModel.from_pretrained(
        config.bert_model, encoder_structure=config.branching_structure)
    else:
      utils.log("Build {}:{} Encoder".format(config.encoder_type, config.bert_model))
      self.encoder = get_encoder(config.encoder_type).from_pretrained(config.bert_model)




    utils.log("Build Task Modules")
    self.tasks_modules = nn.ModuleDict()
    for task in tasks:
      if task.has_module:
        self.tasks_modules.update([(task.name, task.get_module())])
    self.task_dict = dict([(task.name, task) for task in self.tasks])
    self.dummy_input = torch.rand(1, 10, requires_grad=True)

    # self.encoder = HighwayLSTM(num_layers=3, input_size=300, hidden_size=200, layer_dropout=0.2)
    # self.word_embedding = nn.Embedding(self.config.external_vocab_size, self.config.external_vocab_embed_size)
    # self.word_embedding.weight.data.copy_(torch.from_numpy(np.load(config.external_embeddings)))
    # print("Loading embedding from {}".format(config.external_embeddings))

    self.loss_max_margin = MarginRankingLoss(margin=config.max_margin)
    self.distance = nn.PairwiseDistance(p=1)


  def encoding(self, **kwargs):
    task_name = kwargs.get("task_name", None)
    input_ids = kwargs.get("input_ids")
    input_mask = kwargs.get("input_mask")
    segment_ids = kwargs.get("segment_ids")
    extra_args = kwargs.get("extra_args", {})
    output_all_encoded_layers = extra_args.get("output_all_encoded_layers", False)
    route_path = extra_args.get("route_path", None)
    selected_non_final_layers = extra_args.get("selected_non_final_layers", None)
    no_dropout = task_name in READING_COMPREHENSION_TASKS
    langs = kwargs.get("langs", None)

    features = self.encoder(
      input_ids=input_ids,
      token_type_ids=segment_ids,
      attention_mask=input_mask,
      output_all_encoded_layers=output_all_encoded_layers,
      selected_non_final_layers=selected_non_final_layers,
      route_path=route_path,
      no_dropout=no_dropout,
      langs=langs)

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
    task_name = kwargs.pop("task_name")

    outputs_dict = {}

    # parallel teacher student task will use its own outputs_dict
    # It will iterate on each task, and get its output from decoder,
    # and save it on the `results`. The key is the task_name and value
    # is the tensor.
    if task_name == PARALLEL_TEACHER_STUDENT_TASK:
      teacher_prediction = kwargs.pop("teacher_predictions", None)
      if teacher_prediction is None:
        # Run teacher Part
        results = {}
        arguments = self.get_arguments(prefix="a_", kwargs=kwargs)
        features = self.encoding(**arguments)
        for task in self.tasks:
          if task.name != task_name:
            if isinstance(features, list):
              result = features[0]
            else:
              result = features
            # result = self.decoding(**arguments, **kwargs, features=features, task_name=task.name)
            results[task.name] = result.detach()# result.detach()
        return results

      else:
        arguments = self.get_arguments(prefix="b_", kwargs=kwargs)
        features = self.encoding(**arguments)
        losses = {}
        for task in self.tasks:
          if task.name != task_name:
            # result = self.decoding(**arguments, features=features, task_name=task.name)
            if isinstance(features, list):
              result = features[0]
            else:
              result = features
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

    elif task_name in SIAMESE:
      outputs_dict[task_name] = self.siamese_forward(*inputs, **kwargs, task_name=task_name)

    elif task_name == MIXSENT_TASK:
      outputs_dict[task_name] = self.mixsent_forward(*inputs, **kwargs, task_name=task_name)

    elif task_name in TRIPLET:
      outputs_dict[task_name] = self.triplet_forward(*inputs, **kwargs, task_name=task_name)

      # elif task_name in [DEP_PARSING_TASK]:
      #   arguments = self.get_arguments(prefix="", kwargs=kwargs)
      #   features = self.encoding(**arguments)
      #   outputs = self.decoding(**arguments, **kwargs, features=features, task_name=task_name)
      #   outputs_dict[task_name] = outputs

      # elif task_name == ENCODING_TASK:
      #   # For encoding task, for now, we only consider singleton.
      #   arguments = self.get_arguments(prefix="", kwargs=kwargs)
      #   features = self.encoder(**arguments)
      #   return {"features": features}

    else:
      # Currently we can only one-forward-multitask here
      # And the interface is not compatible with dependency parsing and some other multi-outputs task.
      # TODO: Restructure the inference to fix
      arguments = self.get_arguments(prefix="", kwargs=kwargs)
      features = self.encoding(**arguments)
      task_names = task_name.split(',')
      for task_name in task_names:
        logits = self.decoding(**arguments, **kwargs, features=features, task_name=task_name)

        if arguments["label_ids"] is not None:
          if task_name not in SKIP_LOSS_TASK:
          # if task_name in ["joint_srl"]:
          #   loss = self.task_dict[task_name].compute_loss()
          # else:
            loss = get_loss(
              task=self.task_dict[task_name],
              logits=logits,
              config=self.config,
              **arguments, **kwargs)
            outputs_dict[task_name] = {
              "loss": loss,
              "logits": logits}
          else:
            outputs_dict[task_name] = {"logits": logits}
        else:
          if task_name in SKIP_LOSS_TASK:
            outputs_dict[task_name] = {"logits": logits}
          else:
            outputs_dict[task_name] = {"logits": logits.detach()}

    return outputs_dict


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

    return {"loss": loss, "logits": logits}

  def triplet_forward(self, *inputs, **kwargs):
    task_name = kwargs.pop("task_name")
    input_ids = kwargs.pop("input_ids")
    input_mask = kwargs.pop("input_mask")
    segment_ids = kwargs.pop("segment_ids")
    p_input_ids = kwargs.pop("p_input_ids")
    p_input_mask = kwargs.pop("p_input_mask")
    p_segment_ids = kwargs.pop("p_segment_ids")
    n_input_ids = kwargs.pop("n_input_ids")
    n_input_mask = kwargs.pop("n_input_mask")
    n_segment_ids = kwargs.pop("n_segment_ids")

    extra_args = kwargs.pop("extra_args", {})
    output_all_encoded_layers = extra_args.get("output_all_encoded_layers", False)
    route_path = extra_args.get("route_path", None)
    selected_non_final_layers = extra_args.get("selected_non_final_layers", None)

    is_inference = kwargs.pop("is_inference")

    features = self.encoder(
      input_ids=input_ids,
      token_type_ids=segment_ids,
      attention_mask=input_mask,
      output_all_encoded_layers=output_all_encoded_layers,
      selected_non_final_layers=selected_non_final_layers,
      route_path=route_path)

    logits = self.tasks_modules[task_name](features)

    if not is_inference:
      p_features = self.encoder(
        input_ids=p_input_ids,
        token_type_ids=p_segment_ids,
        attention_mask=p_input_mask,
        output_all_encoded_layers=output_all_encoded_layers,
        selected_non_final_layers=selected_non_final_layers,
        route_path=route_path)

      n_features = self.encoder(
        input_ids=n_input_ids,
        token_type_ids=n_segment_ids,
        attention_mask=n_input_mask,
        output_all_encoded_layers=output_all_encoded_layers,
        selected_non_final_layers=selected_non_final_layers,
        route_path=route_path)

      p_logits = self.tasks_modules[task_name](p_features)
      n_logits = self.tasks_modules[task_name](n_features)

      positive_distance = self.distance(logits, p_logits)
      negative_distance = self.distance(logits, n_logits)
      loss = self.loss_max_margin(positive_distance, negative_distance, target= extra_args["target"])

      return {"loss": loss}
    else:
      return {"logits": logits}


  def mixsent_forward(self, *inputs, **kwargs):
    task_name = kwargs.pop("task_name")
    teacher_prediction = kwargs.pop("teacher_predictions", None)
    if teacher_prediction is None:
      # Run teacher Part
      results = {}
      arguments_a = self.get_arguments(prefix="a_", kwargs=kwargs)
      features_a = self.encoding(**arguments_a)
      arguments_b = self.get_arguments(prefix="b_", kwargs=kwargs)
      features_b = self.encoding(**arguments_b)
      for task in self.tasks:
        if task.name != task_name:
          if isinstance(features_a, list):
            features_a = features_a[0]
            features_b = features_b[0]
          # else:
          #   features = features
          result_a = self.decoding(**arguments_a, **kwargs, features=features_a, task_name=task.name)
          result_b = self.decoding(**arguments_b, **kwargs, features=features_b, task_name=task.name)
          # TODO: Detach !!!
          results[task.name] = {
            "a": result_a.detach(),
            "b": result_b.detach()
          }
      return results

    else:
      # Run student and compute the loss
      arguments = self.get_arguments(prefix="c_", kwargs=kwargs)
      features = self.encoding(**arguments)
      losses = {}
      for task in self.tasks:
        if task.name != task_name:
          if isinstance(features, list):
            features = features[0]
          result = self.decoding(**arguments, **kwargs, features=features, task_name=task.name)
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
      return {"loss": reduce(lambda x,y: x+y, losses.values())}