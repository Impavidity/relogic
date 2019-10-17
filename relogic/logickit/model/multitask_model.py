import torch
import torch.nn as nn
from torch.optim import SGD, Adam

from relogic.logickit.base import utils
from relogic.logickit.model.base_model import BaseModel
from relogic.logickit.model.optimization import BertAdam, MultipleOptimizer
import numpy as np
from relogic.logickit.data_io import generate_input
from relogic.logickit.inference import get_inference
from relogic.logickit.dataflow import MiniBatch
from relogic.logickit.utils.utils import entropy

class Model(BaseModel):
  def __init__(self, config, tasks):
    super(Model, self).__init__(config)
    self.tasks = tasks
    utils.log("Building model")
    inference = get_inference(config)(config, tasks)
    utils.log("Switch Model to device")
    inference = inference.to(self.device)
    # TODO: need to test
    if config.multi_gpu:
      inference = torch.nn.DataParallel(inference)
    self.model = inference
    self.teacher = inference
    utils.log(self.model.__str__())

    if config.mode == "train" or config.mode == "finetune":
      self.setup_training(config, tasks)

    ## Inplace Relu
    def inplace_relu(m):
      classname = m.__class__.__name__
      if classname.find('ReLU') != -1:
        m.inplace = True

    inference.apply(inplace_relu)

  def setup_training(self, config, tasks):
    # Calculate optimization steps
    size_train_examples = 0
    if config.mode == "train" or config.mode == "finetune":
      for task in tasks:
        utils.log("{} : {}  training examples".format(task.name, task.train_set.size))
        if "loss_weight" in config.tasks[task.name]:
          utils.log("loss weight {}".format(config.tasks[task.name]["loss_weight"]))
        size_train_examples += task.train_set.size

    config.num_steps_in_one_epoch = size_train_examples // config.train_batch_size
    if config.num_train_optimization_steps == 0:
      config.num_train_optimization_steps = size_train_examples // config.train_batch_size * config.epoch_number \
        if config.schedule_lr else -1
    utils.log("Optimization steps : {}".format(config.num_train_optimization_steps))
    # adjust to real training batch size
    config.train_batch_size = config.train_batch_size // config.gradient_accumulation_steps
    utils.log("Training batch size: {}".format(config.train_batch_size))
    config.test_batch_size = config.test_batch_size // config.gradient_accumulation_steps

    # Optimization
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = list(self.model.named_parameters())
    optimizers = {}
    optim_type = ""
    if config.sep_optim:
      utils.log("Optimizing the module using Adam optimizer ..")
      modules_parameters = [p for n, p in param_optimizer if "bert" not in n]
      bert_optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if ("bert" in n) and (not any(nd in n for nd in no_decay))],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if ("bert" in n) and (any(nd in n for nd in no_decay))],
         'weight_decay': 0.0}
      ]
      optimizers["module_optimizer"] = Adam(params=modules_parameters, lr=config.adam_learning_rate)
      optimizers["bert_optimizer"] = BertAdam(bert_optimizer_grouped_parameters,
                                              lr=config.learning_rate,
                                              warmup=config.warmup_proportion,
                                              schedule=config.schedule_method,
                                              t_total=config.num_train_optimization_steps)
      optim_type = "sep_optim"
    elif config.two_stage_optim:
      utils.log("Optimizing the module with two stage")
      modules_parameters = [p for n, p in param_optimizer if "bert" not in n]
      bert_optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
      optimizers["module_optimizer"] = Adam(params=modules_parameters, lr=config.adam_learning_rate)
      optimizers["bert_optimizer"] = BertAdam(bert_optimizer_grouped_parameters,
                                              lr=config.learning_rate,
                                              warmup=config.warmup_proportion,
                                              schedule=config.schedule_method,
                                              t_total=config.num_train_optimization_steps)
      optim_type = "two_stage_optim"
    elif config.fix_bert:
      utils.log("Optimizing the module using Adam optimizer ..")
      modules_parameters = [p for n, p in param_optimizer if "bert" not in n]
      optimizers["module_optimizer"] = SGD(params=modules_parameters, lr=config.adam_learning_rate)
      optim_type = "fix_bert"
    elif config.fix_embedding:
      utils.log("Optimizing the model using one optimizer and fix embedding layer")
      bert_optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and ("word_embeddings" not in n)],
                'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and ("word_embeddings" not in n)],
         'weight_decay': 0.0}]
      optimizers["bert_optimizer"] = BertAdam(bert_optimizer_grouped_parameters,
                                              lr=config.learning_rate,
                                              warmup=config.warmup_proportion,
                                              schedule=config.schedule_method,
                                              t_total=config.num_train_optimization_steps)
      optim_type = "normal"
    else:
      utils.log("Optimizing the model using one optimizer")
      bert_optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
      optimizers["bert_optimizer"] = BertAdam(bert_optimizer_grouped_parameters,
                                lr=config.learning_rate,
                                warmup=config.warmup_proportion,
                                schedule=config.schedule_method,
                                t_total=config.num_train_optimization_steps)
      optim_type = "normal"

    self.optimizer = MultipleOptimizer(optim=optimizers, optim_type=optim_type)

    self.global_step_labeled = 0
    self.global_step_unlabeled = 0


  def generate_input(self, mb, use_label=True):

    input_ids = torch.tensor([f.input_ids for f in mb.input_features], dtype=torch.long).to(self.device)
    input_mask = torch.tensor([f.input_mask for f in mb.input_features], dtype=torch.long).to(self.device)
    if mb.task_name not in ["squad11", "squad20"]:
      input_head = torch.tensor([f.is_head for f in mb.input_features], dtype=torch.long).to(self.device)
    else:
      input_head = None
    segment_ids = torch.tensor([f.segment_ids for f in mb.input_features], dtype=torch.long).to(self.device)
    if use_label:
      label_ids = torch.tensor([f.label_ids for f in mb.input_features], dtype=torch.long).to(self.device)
    else:
      label_ids = None
    extra_args = {}
    if mb.task_name in ["srl", "srl_conll05", "srl_conll09", "srl_conll12"]:
      is_predicate_id = torch.tensor([f.is_predicate for f in mb.input_features], dtype=torch.long).to(self.device)
      extra_args["is_predicate_id"] = is_predicate_id
    if mb.task_name == 'rel_extraction':
      subj_indicator = torch.tensor([f.subj_indicator for f in mb.input_features], dtype=torch.long).to(self.device)
      obj_indicator = torch.tensor([f.obj_indicator for f in mb.input_features], dtype=torch.long).to(self.device)
      extra_args['subj_indicator'] = subj_indicator
      extra_args['obj_indicator'] = obj_indicator
    if mb.task_name == "predicate_sense":
      temp = torch.tensor([f.label_ids for f in mb.input_features], dtype=torch.long).to(self.device)
      # hard code 'O' == 0 'X' == 22
      extra_args["is_predicate_id"] =  (temp != 0) & (temp != 22)
    if self.config.branching_encoder:
      extra_args["route_path"] = self.config.task_route_paths[mb.task_name]
    return input_ids, input_mask, input_head, segment_ids, label_ids, extra_args

  def flip_the_coin(self, step):
    if np.random.rand() < min(0.6, 1 / 10000 * step):
      return True
    return False

  def gen_masks(self, label_ids):
    sample = torch.rand(label_ids.size())
    return sample

  def train_labeled_abstract(self, mb, step):
    self.model.train()
    if isinstance(mb, MiniBatch):
      inputs = mb.generate_input(device=self.device, use_label=True)
      if "input_ids" in inputs and inputs["input_ids"].size(0) == 0:
        utils.log("Zero Batch")
        return 0
    else:
      if mb.task_name in ["rel_extraction", "srl", "er"]:
        inputs = generate_input(
          mb=mb,
          config=self.config,
          device=self.device)
        if inputs["input_ids"].size(0) == 0:
          utils.log("Zero Batch")
          return 0
      else:
        # TODO: Slow process to change interfaces for all tasks
        inputs = self.generate_input(mb)
        if inputs[0].size(0) == 0:
          utils.log("Zero Batch")
          return 0


    outputs = self.model(**inputs)
    if self.config.output_attentions:
      loss, _, _ = outputs
    else:
      loss, _ = outputs

    loss = mb.loss_weight * loss

    if self.config.gradient_accumulation_steps > 1:
      loss = loss / self.config.gradient_accumulation_steps
    loss.backward()
    if (step + 1) % self.config.gradient_accumulation_steps == 0:
      # TODO: a quick fix
      if not hasattr(mb, "task_name") or mb.task_name not in ["squad11", "squad20"]:
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
      self.optimizer.step()
      self.optimizer.zero_grad()
      self.global_step_labeled += 1
    return loss.item()

  def test_abstract(self, mb):
    self.model.eval()
    if isinstance(mb, MiniBatch):
      inputs = mb.generate_input(self.device, use_label=False)
    else:
      if mb.task_name in ["rel_extraction", "srl", "er"]:
        inputs = generate_input(
          mb=mb,
          config=self.config,
          device=self.device,
          use_label=False)
      else:
        # TODO: Slow process to change interfaces for all tasks
        inputs = self.generate_input(mb, use_label=False)
    with torch.no_grad():
      results = self.model(**inputs)
    if self.config.output_attentions:
      results, attention_map = results
      # list(batch_size, num_heads, sent_length, sent_length) = layer
      attention_map = torch.stack(attention_map, dim=0).transpose(0, 1)
    # (layer, batch, head, length, length)
    if self.config.output_attentions:
      return results, attention_map.cpu().numpy()
    else:
      return results

  def analyze(self, mb, head_mask, params):
    self.model.eval()
    if isinstance(mb, MiniBatch):
      inputs = mb.generate_input(self.device, use_label=False)
    else:
      if mb.task_name in ["rel_extraction", "srl", "er"]:
        inputs = generate_input(
          mb=mb,
          config=self.config,
          device=self.device,
          use_label=True)
      else:
        # TODO: Slow process to change interfaces for all tasks
        inputs = self.generate_input(mb, use_label=False)
    inputs["extra_args"]["head_mask"] = head_mask
    outputs = self.model(**inputs)
    # We assume here is loss, logits, and attention_map
    loss, logits, all_attentions = outputs
    loss.backward()  # Backpropagate to populate the gradients in the head mask
    # Compute Entropy

    for layer, attn in enumerate(all_attentions):
      masked_entropy = entropy(attn.detach()) * inputs["input_mask"].float().unsqueeze(1)
      params["attn_entropy"][layer] += masked_entropy.sum(-1).sum(0).detach()

    params["head_importance"] += head_mask.grad.abs().detach()
    params["total_token"] += inputs["input_mask"].float().detach().sum().data

    return logits, torch.stack(all_attentions, dim=0).transpose(0, 1).detach().data.cpu().numpy()

  def train_labeled(self, mb, step):
    self.model.train()

    input_ids, input_mask, input_head, segment_ids, label_ids, extra_args = self.generate_input(mb)
    if input_ids.size(0) == 0:
      return 0

    if self.config.word_dropout:
      if self.flip_the_coin(self.global_step_labeled):
        sample = self.gen_masks(label_ids)
        input_ids = torch.where(sample.to(self.device) < 0.4, torch.tensor(103, dtype=torch.long).to(self.device) * input_mask, input_ids)

    loss = self.model(
      task_name=mb.task_name,
      input_ids=input_ids,
      input_mask=input_mask,
      input_head=input_head,
      segment_ids=segment_ids,
      label_ids=label_ids,
      extra_args=extra_args)
    if self.config.multi_gpu:
      if loss.size(0) > 1:
        loss = loss.mean()
    # average over multi-gpu
    if self.config.gradient_accumulation_steps > 1:
      loss = loss / self.config.gradient_accumulation_steps
    loss.backward()
    if (step + 1) % self.config.gradient_accumulation_steps == 0:
      if mb.task_name not in ["squad11", "squad20"]:
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
      self.optimizer.step()
      self.optimizer.zero_grad()
      self.global_step_labeled += 1
    return loss.item()


  def train_unlabeled(self, mb, step):
    self.model.train()
    input_ids, input_mask, input_head, segment_ids, label_ids, extra_args = self.generate_input(mb, False)
    teacher_labels = mb.teacher_predictions
    loss = self.model(
        task_name=mb.task_name,
        input_ids=input_ids,
        input_mask=input_mask,
        input_head=input_head,
        segment_ids=segment_ids,
        label_ids=teacher_labels,
        extra_args=extra_args)
    if self.config.gradient_accumulation_steps > 1:
      loss = loss / self.config.gradient_accumulation_steps
    loss.backward()
    if (step + 1) % self.config.gradient_accumulation_steps == 0:
      nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
      self.optimizer.step()
      self.optimizer.zero_grad()
      self.global_step_unlabeled += 1
    return loss.item()

  def run_teacher(self, mb):
    self.teacher.eval()
    input_ids, input_mask, input_head, segment_ids, label_ids, extra_args = self.generate_input(mb, False)
    with torch.no_grad():
      results = self.teacher(
        task_name=mb.task_name,
        input_ids=input_ids,
        input_mask=input_mask,
        input_head=input_head,
        segment_ids=segment_ids,
        label_ids=None,
        extra_args=extra_args)
    for task_name, probs in results.items():
      mb.teacher_predictions[task_name] = probs
      # TODO: check the output format for probs

  def run_teacher_abstract(self, mb: MiniBatch):
    self.teacher.eval()
    inputs = mb.generate_input(device=self.device, use_label=False)
    with torch.no_grad():
      results = self.teacher(**inputs)
      mb.teacher_predictions = results


  def train_unlabeled_abstract(self, mb: MiniBatch, step):
    self.model.train()
    inputs = mb.generate_input(device=self.device, use_label=False)
    outputs = self.model(**inputs,
            teacher_predictions=mb.teacher_predictions)

    loss, _ = outputs

    loss = 0.1 * loss

    if self.config.gradient_accumulation_steps > 1:
      loss = loss / self.config.gradient_accumulation_steps
    loss.backward()
    if (step + 1) % self.config.gradient_accumulation_steps == 0:
      # TODO: a quick fix
      if not hasattr(mb, "task_name") or mb.task_name not in ["squad11", "squad20"]:
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
      self.optimizer.step()
      self.optimizer.zero_grad()
      self.global_step_unlabeled += 1
    return loss.item()


  def test(self, mb):
    self.model.eval()
    input_ids, input_mask, input_head, segment_ids, label_ids, extra_args = self.generate_input(mb, False)
    with torch.no_grad():
      results = self.model(
        task_name=mb.task_name,
        input_ids=input_ids,
        input_mask=input_mask,
        input_head=input_head,
        segment_ids=segment_ids,
        label_ids=None,
        extra_args=extra_args)
    return results
