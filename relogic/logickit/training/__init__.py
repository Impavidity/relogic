# from relogic.logickit.tokenizer.tokenization import BertTokenizer
# from relogic.logickit.model import get_model
# from relogic.logickit.tasks import get_task
# from relogic.logickit.base import utils
# from relogic.logickit.dataflow import DataFlow
# from typing import List
# import torch
# import json
# import os
# import time
# import numpy as np
# import bisect
#
# class Trainer(object):
#   """Trainer class controls the training, validation and evaluation process.
#
#   Args:
#     config (SimpleNamespace): configuration
#
#   """
#   def __init__(self, config):
#     self.config = config
#     self.tokenizer = BertTokenizer.from_pretrained(
#       config.vocab_path, do_lower_case=config.do_lower_case,
#       never_split=config.never_split, lang=config.lang)
#     self.tasks = [
#       get_task(self.config, task_name, self.tokenizer)
#       for task_name in self.config.tasks
#     ]
#     self.model = get_model(config)(config=config, tasks=self.tasks)
#     self.progress = TrainingProgress(config=config)
#
#   def train(self):
#     heading = lambda s: utils.heading(s, '(' + self.config.model_name + ')')
#     trained_on_sentences = 0
#     start_time = time.time()
#     unsupervised_loss_total, unsupervised_loss_count = 0, 0
#     supervised_loss_total, supervised_loss_count = 0, 0
#     step = 0
#
#     for minibatch in self.get_training_minibatches():
#       if minibatch.task_name != "unlabeled":
#         loss = self.model.train_labeled_abstract(minibatch, step)
#         supervised_loss_total += loss
#         supervised_loss_count += 1
#       if minibatch.task_name == 'unlabeled':
#         self.model.run_teacher(minibatch)
#         loss = self.model.train_unlabeled(minibatch, step)
#         unsupervised_loss_total += loss
#         unsupervised_loss_count += 1
#         minibatch.teacher_predictions.clear()
#
#       step += 1
#       trained_on_sentences += minibatch.size
#
#       if self.model.global_step_labeled % self.config.print_every == 0 \
#             and self.model.global_step_unlabeled % self.config.print_every == 0 \
#             and not self.progress.log_in_step(self.model.global_step_labeled):
#         # a quick patch here
#         # TODO: organize better
#         self.model.optimizer.update_loss(supervised_loss_total / max(1, supervised_loss_count))
#
#         utils.log(
#           "step supervised {:} - "
#           "step unsupervised {:} - "
#           "supervised loss: {:.3f} - "
#           "unsupervised loss : {:.3f} - "
#           "{:.1f} sentences per second".format(
#             self.model.global_step_labeled, self.model.global_step_unlabeled,
#             supervised_loss_total / max(1, supervised_loss_count),
#             unsupervised_loss_total / max(1, unsupervised_loss_count),
#             trained_on_sentences / (time.time() - start_time)))
#         unsupervised_loss_total, unsupervised_loss_count = 0, 0
#         supervised_loss_total, supervised_loss_count = 0, 0
#         self.progress.add_log_step(self.model.global_step_labeled)
#
#       if self.model.global_step_labeled % self.config.eval_dev_every == 0 \
#             and self.model.global_step_unlabeled % self.config.eval_dev_every == 0 and \
#             not self.progress.evaluated_in_step(self.model.global_step_labeled):
#         heading("EVAL on DEV")
#         self.evaluate_all_tasks(self.progress.history)
#         self.progress.save_if_best_dev_model(self.model.model)
#         self.progress.add_evaluated_step(self.model.global_step_labeled)
#
#   def evaluate_all_tasks(self, history=None, train_set=False):
#     results = []
#     for task in self.tasks:
#       results.append(self.evaluate_task(task, train_set))
#       if history is not None:
#         results[-1].append(('step', self.model.global_step_labeled))
#     if history is not None:
#       history.append(results)
#     if history is not None:
#       utils.write_pickle(history, self.config.history_file)
#
#   def evaluate_task(self, task, train_set):
#     scorer = task.get_scorer(dump_to_file={"output_dir": self.config.output_dir,
#                                            "task_name": task.name})
#     data = task.train_set if train_set else task.val_set
#     for i, mb in enumerate(data.get_minibatches(self.config.test_batch_size)):
#       # batch_preds = self.model.test(mb)
#       batch_preds = self.model.test_abstract(mb)
#       extra_output = {}
#       if self.config.output_attentions:
#         batch_preds, attention_map = batch_preds
#         extra_output["attention_map"] = attention_map
#       loss = 0
#       scorer.update(mb, batch_preds, loss, extra_output)
#       if i % 100 == 0:
#         utils.log("{} batch processed.".format(i))
#     results = scorer.get_results()
#     utils.log(task.name.upper() + ": " + scorer.results_str())
#     return results
#
#   def restore(self, model_path):
#     restore_state_dict = torch.load(
#       model_path, map_location=lambda storage, location: storage)
#     # loaded_dict = {k: restore_state_dict[k] for k in
#     #                set(self.model.model.state_dict().keys()) & set(restore_state_dict.keys())}
#     # model_state = self.model.model.state_dict()
#     # model_state.update(loaded_dict)
#     for key in self.config.ignore_parameters:
#       # restore_state_dict.pop(key)
#       restore_state_dict[key] = self.model.model.state_dict()[key]
#     self.model.model.load_state_dict(restore_state_dict)
#     utils.log("Model Restored from {}".format(model_path))
#
#   def get_training_minibatches(self):
#     """Generate labeled batches and unlabeled batches
#
#     Returns:
#       Iterator: batch Iterator
#     """
#     datasets: List[DataFlow]  = [task.train_set for task in self.tasks]
#     weights = [np.sqrt(dataset.size) for dataset in datasets]
#     thresholds = np.cumsum([w / np.sum(weights) for w in weights])
#
#     labeled_mbs = [
#       dataset.endless_minibatches(self.config.train_batch_size)
#       for dataset in datasets
#     ]
#     # unlabeled_mbs = unlabeled_data_reader.endless_minibatches(
#     #   self.config.train_batch_size)
#     while True:
#       dataset_ind = bisect.bisect(thresholds, np.random.random())
#       yield next(labeled_mbs[dataset_ind])
#       # TODO: Unsupervised part
#       # if self.config.is_semisup:
#       #   yield next(unlabeled_mbs)
#
#
# class Predictor(object):
#   """Predictor class basically replicate the trainer, but a simplified version.
#
#   This is mainly for deployment.
#
#   Args:
#     config (SimpleNamespace): configuration
#
#   """
#   def __init__(self):
#     pass
#
# class TrainingProgress(object):
#   """TrainingProgress keeps the logs (loss, validation performance) during the
#   training process.
#
#   TODO: The TrainingProgress is also used to reload the model,
#     recover the unlabeled data checking point.
#
#   Args:
#     config (SimpleNamespace): configuration.
#
#   """
#   def __init__(self, config, restore_if_possible=True):
#     self.config = config
#     if restore_if_possible and os.path.exists(config.progress):
#       history, current_file, current_line = utils.load_pickle(
#         config.progress, memoized=False)
#       self.history = history
#     else:
#       utils.log("No previous checkpoint found - starting from scratch")
#       self.history = []
#     self.evaluated_steps = set([0])
#     # At step 0, the evaluation is not required.
#     self.log_steps = set([])
#
#   def save_if_best_dev_model(self, model):
#     """Save the best model.
#
#     TODO: optimize the code here
#     """
#     best_avg_score = 0
#     for i, results in enumerate(self.history):
#       for result in results:
#         if any("train" in metric for metric, value in result):
#           continue
#         if any("test" in metric for metric, value in result):
#           continue
#       total, count = 0, 0
#       for result in results:
#         for metric, value in result:
#           if "f1" in metric or "las" in metric or "accuracy" in metric or "recall_left" in metric or "recall_right" in metric:
#             total += value
#             count += 1
#       avg_score = total / count
#       if avg_score >= best_avg_score:
#         best_avg_score = avg_score
#         if i == len(self.history) - 1:
#           utils.log("New Score {}, New best model! Saving ...".format(best_avg_score))
#           torch.save(model.state_dict(), os.path.join(self.config.output_dir, self.config.model_name + ".ckpt"))
#           general_config_path = os.path.join(self.config.output_dir, "general_config.json")
#           with open(general_config_path, "w") as fout:
#             fout.write(json.dumps(vars(self.config)))
#           # TODO: finish model saving
#
#   def evaluated_in_step(self, step):
#     return step in self.evaluated_steps
#
#   def add_evaluated_step(self, step):
#     self.evaluated_steps.add(step)
#
#   def log_in_step(self, step):
#     return step in self.log_steps
#
#   def add_log_step(self, step):
#     self.log_steps.add(step)