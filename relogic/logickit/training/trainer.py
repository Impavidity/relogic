from __future__ import absolute_import, division, print_function

import bisect
import time
from relogic.logickit.training.training_scheme import TRAINING_SCHEME
import numpy as np
import json

import torch
from relogic.logickit.base import utils
from relogic.logickit.dataflow import DataFlow
from relogic.logickit.tokenizer.tokenization import BertTokenizer
from relogic.logickit.tokenizer.fasttext_tokenization import FasttextTokenizer
from relogic.logickit.tasks import get_task
from relogic.logickit.model import get_model
from relogic.logickit.training.training_progress import TrainingProgress
from relogic.logickit.dataflow import MiniBatch
import os
from relogic.logickit.utils.utils import print_2d_tensor
from relogic.logickit.base.constants import ECP_TASK, IR_TASK, NER_TASK, PARALLEL_MAPPING_TASK, DISTILL_TASKS, PARALLEL_TEACHER_STUDENT_TASK
from relogic.logickit.base.configuration_trainer import TrainerConfig
from relogic.logickit.tokenizer import NAME_TO_TOKENIZER_MAP

class Trainer(object):
  def __init__(self, config, teacher_config=None,
        tokenizers = None,
        model = None):
    """
    TODO: We are going to change the interface a little bit for easier configuration, less hard-coded.
    :param config:
    :param teacher_config:
    :param tokenizers:
    :param model:
    """
    self.config = config

    trainer_config : TrainerConfig = TrainerConfig.load_from_json_file(config.trainer_config)

    if tokenizers is not None:
      self.tokenizers = tokenizers
    else:
      self.tokenizers = {}
      for tokenizer_name, params in trainer_config.tokenizers.items():
        self.tokenizers[tokenizer_name] = NAME_TO_TOKENIZER_MAP[tokenizer_name].from_pretrained(**params)

    # A quick fix for version migration
    self.tasks = [
      get_task(self.config, task_name, self.tokenizers)
      for task_name in self.config.task_names
    ]
    self.model = get_model(config)(config=self.config, tasks=self.tasks)

    if self.config.use_external_teacher:
      # Extension for multi-model distillation
      self.teacher_tasks = [
        get_task(teacher_config, task_name, self.tokenizers)
        for task_name in teacher_config.task_names
      ]
      self.teacher_model = get_model(teacher_config)(config=teacher_config, tasks=self.teacher_tasks)

  def train(self, progress: TrainingProgress):
    heading = lambda s: utils.heading(s, '(' + self.config.model_name + ')')
    trained_on_sentences = 0
    start_time = time.time()
    unsupervised_loss_total, unsupervised_loss_count = 0, 0
    supervised_loss_total, supervised_loss_count = 0, 0
    step = 0
    # self.evaluate_all_tasks(progress.history)
    for mb in self.get_training_mbs():
      if isinstance(mb, MiniBatch):
        if mb.task_name not in DISTILL_TASKS:
          loss = self.model.train_labeled_abstract(mb, step)
          supervised_loss_total += loss
          supervised_loss_count += 1
        else:
          if self.config.use_external_teacher:
            self.teacher_model.run_teacher_abstract(mb)
          else:
            self.model.run_teacher_abstract(mb)
          loss = self.model.train_unlabeled_abstract(mb, step)
          unsupervised_loss_total += loss
          unsupervised_loss_count += 1
          mb.teacher_predictions.clear()
      else:
        if mb.task_name != "unlabeled":
          loss = self.model.train_labeled_abstract(mb, step)
          supervised_loss_total += loss
          supervised_loss_count += 1
        if mb.task_name == 'unlabeled':
          self.model.run_teacher(mb)
          loss = self.model.train_unlabeled(mb, step)
          unsupervised_loss_total += loss
          unsupervised_loss_count += 1
          mb.teacher_predictions.clear()

      step += 1
      trained_on_sentences += mb.size


      if self.model.global_step_labeled % self.config.print_every == 0 \
            and not progress.log_in_step(self.model.global_step_labeled):
              # and self.model.global_step_unlabeled % self.config.print_every == 0 \

        # a quick patch here
        # TODO: organize better
        self.model.optimizer.update_loss(supervised_loss_total / max(1, supervised_loss_count))

        utils.log(
          "step supervised {:} - "
          "step unsupervised {:} - "
          "supervised loss: {:.3f} - "
          "unsupervised loss : {:.3f} - "
          "{:.1f} sentences per second".format(
            self.model.global_step_labeled, self.model.global_step_unlabeled,
            supervised_loss_total / max(1, supervised_loss_count),
            unsupervised_loss_total / max(1, unsupervised_loss_count),
            trained_on_sentences / (time.time() - start_time)))
        unsupervised_loss_total, unsupervised_loss_count = 0, 0
        supervised_loss_total, supervised_loss_count = 0, 0
        progress.add_log_step(self.model.global_step_labeled)

      if self.model.global_step_labeled % self.config.eval_dev_every == 0 \
            and not progress.evaluated_in_step(self.model.global_step_labeled):
            # and self.model.global_step_unlabeled % self.config.eval_dev_every == 0 and \

        heading("EVAL on DEV")
        self.evaluate_all_tasks(progress.history)
        progress.save_if_best_dev_model(self.model.model)
        progress.add_evaluated_step(self.model.global_step_labeled)

      if self.config.early_stop_at > 0 and self.model.global_step_labeled >= self.config.early_stop_at:
        utils.log("Early stop at step {}".format(self.model.global_step_labeled))
        break

  def evaluate_all_tasks(self, history=None, train_set=False):
    results = []
    for task in self.tasks:
      results.append(self.evaluate_task(task, train_set))
      if history is not None:
        results[-1].append(('step', self.model.global_step_labeled))
    if history is not None:
      history.append(results)
    if history is not None:
      utils.write_pickle(history, self.config.history_file)

  def evaluate_task(self, task, train_set):
    scorer = task.get_scorer(dump_to_file={"output_dir": self.config.output_dir,
                                           "task_name": task.name})
    data: DataFlow = task.train_set if train_set else task.val_set
    for i, mb in enumerate(data.get_minibatches(self.config.tasks[data.task_name]["test_batch_size"])):
      # batch_preds = self.model.test(mb)
      batch_preds = self.model.test_abstract(mb)

      if isinstance(batch_preds, dict):
        scorer.update(mb, batch_preds, 0, None)
      else:
        # Slow Migration towards the return interface
        extra_output = {}
        if self.config.output_attentions:
          batch_preds, attention_map = batch_preds
          extra_output["attention_map"] = attention_map
        if isinstance(batch_preds, tuple):
          loss, batch_preds = batch_preds
        else:
          loss = 0
        scorer.update(mb, batch_preds, loss, extra_output)
      if i % 100 == 0:
        utils.log("{} batch processed.".format(i))
    results = scorer.get_results()
    utils.log(task.name.upper() + ": " + scorer.results_str())
    return results

  def analyze_task(self, task, head_mask,
                   head_importance, attn_entropy):
    scorer = task.get_scorer(dump_to_file={"output_dir": self.config.output_dir,
                                           "task_name": task.name})
    params = {
      "head_importance": head_importance,
      "attn_entropy": attn_entropy,
      "total_token": 0.0
    }
    data : DataFlow = task.val_set
    # The output of the model is logits and attention_map
    for i, mb in enumerate(data.get_minibatches(self.config.tasks[data.task_name]["test_batch_size"])):
      batch_preds, attention_map = self.model.analyze(mb, head_mask, params=params)
      extra_output = {}
      extra_output["attention_map"] = attention_map
      loss = 0
      scorer.update(mb, batch_preds, loss, extra_output)
      if i % 100 == 0:
        utils.log("{} batch processed.".format(i))
    results = scorer.get_results()
    utils.log(task.name.upper() + ": " + scorer.results_str())

    params["attn_entropy"] /= params["total_token"]
    params["head_importance"] /= params["total_token"]
    np.save(os.path.join(self.config.output_dir, 'attn_entropy.npy'), attn_entropy.detach().cpu().numpy())
    np.save(os.path.join(self.config.output_dir, 'head_importance.npy'), head_importance.detach().cpu().numpy())

    utils.log("Attention entropies")
    print_2d_tensor(attn_entropy)
    utils.log("Head importance scores")
    print_2d_tensor(head_importance)
    utils.log("Head ranked by importance scores")
    head_ranks = torch.zeros(head_importance.numel(), dtype=torch.long, device=self.model.device)
    head_ranks[head_importance.view(-1).sort(descending=True)[1]] = torch.arange(head_importance.numel(),
                                                                                 device=self.model.device)
    head_ranks = head_ranks.view_as(head_importance)
    print_2d_tensor(head_ranks)
    return results

  def get_training_mbs(self):
    if self.config.training_scheme is not None:
      yield from TRAINING_SCHEME[self.config.training_scheme](self.config, self.tasks)
    else:
      datasets = [task.train_set for task in self.tasks]
      weights = [np.sqrt(dataset.size) for dataset in datasets]
      thresholds = np.cumsum([w / np.sum(weights) for w in weights])

      labeled_mbs = [
        dataset.endless_minibatches(self.config.tasks[dataset.task_name]["train_batch_size"])
        for dataset in datasets
      ]
      while True:
        dataset_ind = bisect.bisect(thresholds, np.random.random())
        yield next(labeled_mbs[dataset_ind])


  def restore(self, model_path):
    restore_state_dict = torch.load(
      model_path, map_location=lambda storage, location: storage)
    # loaded_dict = {k: restore_state_dict[k] for k in
    #                set(self.model.model.state_dict().keys()) & set(restore_state_dict.keys())}
    # model_state = self.model.model.state_dict()
    # model_state.update(loaded_dict)
    for key in self.config.ignore_parameters:
      # restore_state_dict.pop(key)
      restore_state_dict[key] = self.model.model.state_dict()[key]
    self.model.model.load_state_dict(restore_state_dict)
    utils.log("Model Restored from {}".format(model_path))

  def restore_teacher(self, model_path):
    restore_state_dict = torch.load(
      model_path, map_location=lambda storage, location: storage)
    # loaded_dict = {k: restore_state_dict[k] for k in
    #                set(self.model.model.state_dict().keys()) & set(restore_state_dict.keys())}
    # model_state = self.model.model.state_dict()
    # model_state.update(loaded_dict)
    for key in self.config.ignore_parameters:
      # restore_state_dict.pop(key)
      restore_state_dict[key] = self.model.model.state_dict()[key]
    self.teacher_model.model.load_state_dict(restore_state_dict)
    utils.log("Teacher Model Restored from {}".format(model_path))
