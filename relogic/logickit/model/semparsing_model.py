from relogic.logickit.model.base_model import BaseModel
from relogic.logickit.base.configuration import Configuration
from relogic.logickit.dataflow import MiniBatch
import torch
import torch.nn as nn
from collections import defaultdict

from relogic.logickit.base import utils
from relogic.logickit.model.optimization import BertAdam, MultipleOptimizer
# from relogic.logickit.modules.semparse.column_selection import BertForColumnSelection
from relogic.logickit.modules.semparse.RAT import RAT
from relogic.logickit.modules.semparse.joint_ct import JointCTRAT
from relogic.logickit.modules.semparse.EditNet import EditNet
from relogic.logickit.modules.semparse.slot_filling import SlotFilling
from relogic.logickit.modules.semparse.reranker import Reranker
from relogic.logickit.modules.semparse.zh_joint_ct import ZHJointCTSQL
import math
from torch import autograd
from relogic.logickit.base.constants import ZH_JOINT_CT_SQL_TASK, SLOT_FILLING_SQL_TASK, EDITNET_SQL_TASK, SQL_RERANKING_TASK
from transformers.optimization import AdamW, get_linear_schedule_with_warmup


class SemParsingModel(BaseModel):
  def __init__(self, config, tasks, ext_config: Configuration):
    super().__init__(config, ext_config)
    self.tasks = tasks
    self.config = config

    # self.model = BertForColumnSelection(config=config, task_name=self.tasks[0].name, n_classes=self.tasks[0].n_classes)
    if config.task_names[0] == "rat_sql":
      self.model = RAT(config=config, label_mapping=self.tasks[0].loader.label_mapping)
    if config.task_names[0] == "joint_ct_rat_sql":
      self.model = JointCTRAT(config=config, label_mapping=self.tasks[0].loader.label_mapping)
    if config.task_names[0] == SLOT_FILLING_SQL_TASK:
      self.model = SlotFilling(config=config)
    if config.task_names[0] == EDITNET_SQL_TASK:
      self.model = EditNet(config=config)
    if config.task_names[0] == SQL_RERANKING_TASK:
      self.model = Reranker(config=config)
    if config.task_names[0] == ZH_JOINT_CT_SQL_TASK:
      self.model = ZHJointCTSQL(config=config)
    self.model.to(self.device)


    self.setup_training(config, tasks)

  def setup_training(self, config, tasks):
    super().setup_training(config, tasks)
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    if not hasattr(config, "use_adamw") or not config.use_adamw:
      param_optimizer = list(self.model.named_parameters())
      #
      # optimizers = {}
      # optim_type = ""
      #
      # # For now I will directly use the BertAdam for optimizing the models.
      # # Later will the model become complicated, we can further change this.
      # utils.log("Optimizing the model using one optimizer")
      optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in ['bert'])], 'lr': config.learning_rate},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in ['bert'])], 'lr': config.learning_rate * 0.05}]
      # optimizers["bert_optimizer"] = BertAdam(optimizer_grouped_parameters,
      #                                         lr=config.learning_rate,
      #                                         warmup=config.warmup_proportion,
      #                                         schedule=config.schedule_method,
      #                                         t_total=config.num_train_optimization_steps)
      # self.optimizer = MultipleOptimizer(optim=optimizers, optim_type=optim_type)


      optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=config.learning_rate)

      if config.use_lr_scheduler:
        print("Enable Learning Rate Scheduler")
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[21, 41],
                                                              gamma=config.lr_scheduler_gammar)
      else:
        self.scheduler = None
    else:
      t_total = config.num_train_optimization_steps
      # We set to optimization step manually
      no_decay = ["bias", "LayerNorm.weight"]
      optimizer_grouped_parameters = [
        {
          "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
          "weight_decay": 0.01,
        },
        {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
      ]
      optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
      self.scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=t_total
      )



    self.optimizer = MultipleOptimizer(optim={"adam": optimizer}, optim_type="")

    self.global_step_labeled = 0
    self.global_step_unlabeled = 0
    self.log_counter = 0
    self.log_dict = defaultdict(float)
    self.use_pseudo_count = 0


  def train_labeled_abstract(self, mb: MiniBatch, step):


    self.model.train()


    inputs = mb.generate_input(device=self.device, use_label=True)

    inputs["step"] = self.global_step_labeled

    outputs = self.model(**inputs)
    if outputs is None:
      return 0

    # We want to standardize the interface, so we keep mb.task_name for the outputs
    loss = outputs[mb.task_name].pop("loss")
    if outputs[mb.task_name]["use_pseudo"]:
      self.use_pseudo_count += 1

    loss = mb.loss_weight * loss

    if self.config.gradient_accumulation_steps > 1:
      loss = loss / self.config.gradient_accumulation_steps
    loss.backward()

    if (step + 1) % self.config.gradient_accumulation_steps == 0:
      total_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

      self.optimizer.step()
      if hasattr(self.config, "use_adamw") and self.config.use_adamw:
        self.scheduler.step()
      self.optimizer.zero_grad()
      self.global_step_labeled += 1

      # Do the lr_scheduler if necessary

      if not self.config.use_adamw and self.global_step_labeled % self.config.num_steps_in_one_epoch == 0:
        if self.config.use_lr_scheduler:
          self.scheduler.step()
          print("lr", self.scheduler.get_lr()[0])

      if self.global_step_labeled % self.config.print_every == 0:
        print("Pseudo Count: {:} Ratio {:.3f} -".format(self.use_pseudo_count,
                                                      self.use_pseudo_count / (self.global_step_labeled + 1)), end=" ")

    for key, value in outputs[mb.task_name].items():
      self.log_dict[f"{mb.task_name}/{key}"] += value
    self.log_counter += 1


    if self.log_counter == self.config.print_every:
      for key, value in self.log_dict.items():
        self.tb_writer.add_scalar(key, value / self.log_counter, self.global_step_labeled)
      self.log_counter = 0
      self.log_dict = defaultdict(float)

    return loss.item() * self.config.gradient_accumulation_steps

  def test_abstract(self, mb):
    self.model.eval()

    inputs = mb.generate_input(self.device, use_label=False)

    with torch.no_grad():
      results = self.model(**inputs)
      return results


