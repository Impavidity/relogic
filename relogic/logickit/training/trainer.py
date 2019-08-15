from __future__ import absolute_import, division, print_function

import bisect
import time

import numpy as np

import torch
from relogic.logickit.base import utils

from relogic.logickit.tokenizer.tokenization import BertTokenizer
from relogic.logickit.tasks import get_task
from relogic.logickit.model import get_model
from relogic.logickit.training.training_progress import TrainingProgress



class Trainer(object):
  def __init__(self, config):
    self.config = config
    self.tokenizer = BertTokenizer.from_pretrained(
      config.vocab_path, do_lower_case=config.do_lower_case,
      never_split=config.never_split, lang=config.lang)
    self.tasks = [
      get_task(self.config, task_name, self.tokenizer)
      for task_name in self.config.task_names
    ]
    self.model = get_model(config)(config=self.config, tasks=self.tasks)


  def train(self, progress: TrainingProgress):
    heading = lambda s: utils.heading(s, '(' + self.config.model_name + ')')
    trained_on_sentences = 0
    start_time = time.time()
    unsupervised_loss_total, unsupervised_loss_count = 0, 0
    supervised_loss_total, supervised_loss_count = 0, 0
    step = 0
    # self.evaluate_all_tasks(progress.history)

    for mb in self.get_training_mbs(progress.unlabeled_data_reader):
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
            and self.model.global_step_unlabeled % self.config.print_every == 0 \
            and not progress.log_in_step(self.model.global_step_labeled):
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
            and self.model.global_step_unlabeled % self.config.eval_dev_every == 0 and \
            not progress.evaluated_in_step(self.model.global_step_labeled):
        heading("EVAL on DEV")
        self.evaluate_all_tasks(progress.history)
        progress.save_if_best_dev_model(self.model.model)
        progress.add_evaluated_step(self.model.global_step_labeled)

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
    data = task.train_set if train_set else task.val_set
    for i, mb in enumerate(data.get_minibatches(self.config.test_batch_size)):
      # batch_preds = self.model.test(mb)
      batch_preds = self.model.test_abstract(mb)
      extra_output = {}
      if self.config.output_attentions:
        batch_preds, attention_map = batch_preds
        extra_output["attention_map"] = attention_map
      loss = 0
      scorer.update(mb, batch_preds, loss, extra_output)
      if i % 100 == 0:
        utils.log("{} batch processed.".format(i))
    results = scorer.get_results()
    utils.log(task.name.upper() + ": " + scorer.results_str())
    return results

  def get_training_mbs(self, unlabeled_data_reader):
    datasets = [task.train_set for task in self.tasks]
    weights = [np.sqrt(dataset.size) for dataset in datasets]
    thresholds = np.cumsum([w / np.sum(weights) for w in weights])

    labeled_mbs = [
      dataset.endless_minibatches(self.config.train_batch_size)
      for dataset in datasets
    ]
    unlabeled_mbs = unlabeled_data_reader.endless_minibatches(
      self.config.train_batch_size)
    while True:
      dataset_ind = bisect.bisect(thresholds, np.random.random())
      yield next(labeled_mbs[dataset_ind])
      if self.config.is_semisup:
        yield next(unlabeled_mbs)

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

  def predict(self, inputs, task_name):
    # First, according the inputs to create examples, features, and batch
    batch = []
    scorer = self.tasks[task_name].get_scorer()
    # Need to fix the get scorer.
    # Basically you will recreate a scorer for one `predict` call
    # You need to reuse
    for i, mb in enumerate(batch):
      batch_preds = self.model.test_abstract(mb)
      scorer.update(batch_preds)
