import torch
import torch.nn as nn

from relogic.logickit.base import utils
from relogic.logickit.model.optimization import BertAdam
from relogic.logickit.inference.retire_inference import PairMatching

class PairMatchingModel(object):
  def __init__(self, config, tasks):
    self.config = config

    if config.local_rank == -1 or config.no_cuda:
      self.device = torch.device("cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu")
      n_gpu = torch.cuda.device_count()
    else:
      torch.cuda.set_device(config.local_rank)
      self.device = torch.device("cuda:" + str(config.local_rank))
      n_gpu = 1
    utils.log("device: {}".format(self.device))

    utils.log("Building model")
    inference = PairMatching(config, tasks)
    utils.log("Switch Model to device")
    inference = inference.to(self.device)
    self.model = inference
    utils.log(self.model.__str__())

    # Optimization
    param_optimizer = list(self.model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
      {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
      {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    size_train_examples = 0
    if config.mode == "train":
      for task in tasks:
        utils.log("{} : {}  training examples".format(task.name, task.train_set.size))
        size_train_examples += task.train_set.size

    config.num_train_optimization_steps = size_train_examples / config.train_batch_size * config.epoch_number \
      if config.schedule_lr else -1
    utils.log("Optimization steps : {}".format(config.num_train_optimization_steps))
    self.optimizer = BertAdam(optimizer_grouped_parameters,
                              lr=config.learning_rate,
                              warmup=config.warmup_proportion,
                              schedule=config.schedule_method,
                              t_total=config.num_train_optimization_steps)

    self.global_step_labeled = 0
    self.global_step_unlabeled = 0

  def generate_pair_input(self, mb):
    a_input_ids = torch.tensor([f.a_input_ids for f in mb.input_features], dtype=torch.long).to(self.device)
    a_input_mask = torch.tensor([f.a_input_mask for f in mb.input_features], dtype=torch.long).to(self.device)
    a_input_head = torch.tensor([f.a_is_head for f in mb.input_features], dtype=torch.long).to(self.device)
    a_segment_ids = torch.tensor([f.a_segment_ids for f in mb.input_features], dtype=torch.long).to(self.device)
    b_input_ids = torch.tensor([f.b_input_ids for f in mb.input_features], dtype=torch.long).to(self.device)
    b_input_mask = torch.tensor([f.b_input_mask for f in mb.input_features], dtype=torch.long).to(self.device)
    b_input_head = torch.tensor([f.b_is_head for f in mb.input_features], dtype=torch.long).to(self.device)
    b_segment_ids = torch.tensor([f.b_segment_ids for f in mb.input_features], dtype=torch.long).to(self.device)
    c_input_ids = torch.tensor([f.c_input_ids for f in mb.input_features], dtype=torch.long).to(self.device)
    c_input_mask = torch.tensor([f.c_input_mask for f in mb.input_features], dtype=torch.long).to(self.device)
    c_input_head = torch.tensor([f.c_is_head for f in mb.input_features], dtype=torch.long).to(self.device)
    c_segment_ids = torch.tensor([f.c_segment_ids for f in mb.input_features], dtype=torch.long).to(self.device)
    extra_args = {"target": -torch.ones(a_input_ids.size(0)).to(self.device)}
    return a_input_ids, a_input_mask, a_input_head, a_segment_ids, \
           b_input_ids, b_input_mask, b_input_head, b_segment_ids, \
           c_input_ids, c_input_mask, c_input_head, c_segment_ids, extra_args

  def generate_singleton_input(self, mb):
    input_ids = torch.tensor([f.input_ids for f in mb.input_features], dtype=torch.long).to(self.device)
    input_mask = torch.tensor([f.input_mask for f in mb.input_features], dtype=torch.long).to(self.device)
    input_head = torch.tensor([f.is_head for f in mb.input_features], dtype=torch.long).to(self.device)
    segment_ids = torch.tensor([f.segment_ids for f in mb.input_features], dtype=torch.long).to(self.device)
    extra_args = {}
    return input_ids, input_mask, input_head, segment_ids, extra_args


  def train_labeled(self, mb, step):
    self.model.train()

    a_input_ids, a_input_mask, a_input_head, a_segment_ids, \
    b_input_ids, b_input_mask, b_input_head, b_segment_ids, \
    c_input_ids, c_input_mask, c_input_head, c_segment_ids, extra_args = self.generate_pair_input(mb)


    loss = self.model(
      task_name = mb.task_name,
      a_input_ids=a_input_ids,
      a_input_mask=a_input_mask,
      a_input_head=a_input_head,
      a_segment_ids=a_segment_ids,
      b_input_ids=b_input_ids,
      b_input_mask=b_input_mask,
      b_input_head=b_input_head,
      b_segment_ids=b_segment_ids,
      c_input_ids=c_input_ids,
      c_input_mask=c_input_mask,
      c_input_head=c_input_head,
      c_segment_ids=c_segment_ids,
      extra_args=extra_args)
    if self.config.gradient_accumulation_steps > 1:
      loss = loss / self.config.gradient_accumulation_steps
    loss.backward()
    if (step + 1) % self.config.gradient_accumulation_steps == 0:
      nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
      self.optimizer.step()
      self.optimizer.zero_grad()
      self.global_step_labeled += 1
    return loss.item()

  def test(self, mb):
    self.model.eval()

    input_ids, input_mask, input_head, segment_ids, extra_args = self.generate_singleton_input(mb)

    with torch.no_grad():
      representation = self.model.gen_repr(
        task_name=mb.task_name,
        input_ids=input_ids,
        input_mask=input_mask,
        input_head=input_head,
        segment_ids=segment_ids,
        extra_args=extra_args)

    return representation



