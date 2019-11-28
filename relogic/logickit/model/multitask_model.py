import torch
import torch.nn as nn
from torch.optim import SGD, Adam

from relogic.logickit.base import utils
from relogic.logickit.model.base_model import BaseModel
from relogic.logickit.model.optimization import BertAdam, MultipleOptimizer
import numpy as np
from relogic.logickit.inference import get_inference
from relogic.logickit.dataflow import MiniBatch
from relogic.logickit.utils.utils import entropy
from relogic.logickit.base.configuration import Configuration

class Model(BaseModel):
  def __init__(self, config, tasks, ext_config=None):
    super(Model, self).__init__(config, ext_config)
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
    config.num_steps_in_one_epoch = 0
    if config.mode == "train" or config.mode == "finetune":
      for task in tasks:
        utils.log("{} : {}  training examples".format(task.name, task.train_set.size))
        if "loss_weight" in config.tasks[task.name]:
          utils.log("loss weight {}".format(config.tasks[task.name]["loss_weight"]))
        size_train_examples += task.train_set.size
        config.num_steps_in_one_epoch += task.train_set.size // config.tasks[task.name]["train_batch_size"]

        # config.train_batch_size = config.train_batch_size // config.gradient_accumulation_steps
        # config.test_batch_size = config.test_batch_size // config.gradient_accumulation_steps
        config.tasks[task.name]["train_batch_size"] =  config.tasks[task.name]["train_batch_size"] // config.gradient_accumulation_steps
        config.tasks[task.name]["test_batch_size"] = config.tasks[task.name]["test_batch_size"] // config.gradient_accumulation_steps
        # adjust to real training batch size
        utils.log("Training batch size: {}".format(config.tasks[task.name]["train_batch_size"]))

    if config.num_train_optimization_steps == 0:
      config.num_train_optimization_steps = config.num_steps_in_one_epoch * config.epoch_number \
        if config.schedule_lr else -1
    utils.log("Optimization steps : {}".format(config.num_train_optimization_steps))

    # Optimization
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = list(self.model.named_parameters())
    # for n, p in param_optimizer:
    #   print(n)
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
    elif self.ext_config.encoder_configs.fix_embedding or self.ext_config.encoder_configs.fix_layers:
      utils.log("Fixing layers from config")
      print("Fix embedding {}".format(self.ext_config.encoder_configs.fix_embedding))
      print("Fix layers {}".format(self.ext_config.encoder_configs.fix_layers))
      self.ext_config: Configuration
      if config.encoder_type == 'xlmr':
        prefix = "layers"
        embed_prefix = "embed_tokens"
      elif config.encoder_type == "bert":
        prefix = "layer"
        embed_prefix = "word_embeddings"
      else:
        raise ValueError("Not supported encoder_type {}".format(config.encoder_type))
      bert_optimizer_grouped_parameters = [
        {'params': [],
         'weight_decay': 0.01},
        {'params': [],
        'weight_decay': 0.0}]

      for n, p in param_optimizer:
        if any(nd in n for nd in no_decay):
          # Checking embedding
          if embed_prefix in n:
            if not self.ext_config.encoder_configs.fix_embedding:
              bert_optimizer_grouped_parameters[1]['params'].append(p)
            else:
              print("Skip {}".format(n))
          if not any(".{}.{}.".format(prefix, l) in n for l in self.ext_config.encoder_configs.fix_layers):
            bert_optimizer_grouped_parameters[1]['params'].append(p)
          else:
            print("Skip {}".format(n))

        else:
          if embed_prefix in n:
            if not self.ext_config.encoder_configs.fix_embedding:
              bert_optimizer_grouped_parameters[0]['params'].append(p)
            else:
              print("Skip {}".format(n))
          if not any(".{}.{}.".format(prefix, l) in n for l in self.ext_config.encoder_configs.fix_layers):
            bert_optimizer_grouped_parameters[0]['params'].append(p)
          else:
            print("Skip {}".format(n))

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


  def flip_the_coin(self, step):
    if np.random.rand() < min(0.6, 1 / 10000 * step):
      return True
    return False

  def gen_masks(self, label_ids):
    sample = torch.rand(label_ids.size())
    return sample

  def train_labeled_abstract(self, mb, step):
    self.model.train()

    inputs = mb.generate_input(device=self.device, use_label=True)
    if "input_ids" in inputs and inputs["input_ids"].size(0) == 0:
      utils.log("Zero Batch")
      return 0

    outputs = self.model(**inputs)

    # TODO: Slow process Migrating Interface ...
    if isinstance(outputs, dict):
      loss = outputs["loss"]
    else:
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

    inputs = mb.generate_input(self.device, use_label=False)

    with torch.no_grad():
      results = self.model(**inputs)
    if isinstance(results, dict):
      return results
    else:
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

    inputs = mb.generate_input(self.device, use_label=False)

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
      self.global_step_unlabeled += 1
    return loss.item()