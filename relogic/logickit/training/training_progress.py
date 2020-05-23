from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from relogic.logickit.base import utils
import torch
import json

class TrainingProgress(object):
  """
  TODO: We are going to retire the Training Progress somehow.
  1. Remove the unlabeled data. The unlabeled data interface will be integrated into
     dataflow.

  """
  def __init__(self, config, restore_if_possible=True):
    self.config = config
    if restore_if_possible and os.path.exists(config.progress):
      history, current_file, current_line = utils.load_pickle(
        config.progress, memoized=False)
      self.history = history
      # self.unlabeled_data_reader =
    else:
      utils.log("No previous checkpoint found - starting from scratch")
      self.history = []
    self.evaluated_steps = set([0])
    self.log_steps = set([])
    self.save_counter = 0
    # We do not want to evaluate in step 0

  def write(self):
    utils.write_pickle(
      (self.history, self.unlabeled_data_reader.current_file,
       self.unlabeled_data_reader.current_line),
      self.config.progress)

  def save_if_best_dev_model(self, model):
    # Why it is average score here
    # TODO: double check the results format
    best_avg_score = 0
    for i, results in enumerate(self.history):
      for result in results:
        if any("train" in metric for metric, value in result):
          continue
        if any("test" in metric for metric, value in result):
          continue
      total, count = 0, 0
      for result in results:
        for metric, value in result:
          if hasattr(self.config, "metrics") and self.config.metrics is not None:
            if metric in self.config.metrics:
              total += value
              count += 1
          elif "distance" in metric or "f1" in metric or "las" in metric or "accuracy" in metric or "recall_left" in metric or "recall_right" in metric or "map" in metric:
            total += value
            count += 1
      avg_score = total / count
      if avg_score >= best_avg_score:
        best_avg_score = avg_score
        if i == len(self.history) - 1:
          model_path = os.path.join(self.config.output_dir, self.config.model_name + "_{}.ckpt".format(self.save_counter))
          utils.log("New Score {}, New best model! Saving to {}...".format(best_avg_score, model_path))
          torch.save(model.state_dict(), model_path)
          general_config_path = os.path.join(self.config.output_dir, "general_config.json")
          with open(general_config_path, "w") as fout:
            fout.write(json.dumps(vars(self.config)))
          # TODO: finish model saving
          self.save_counter += 1

  def evaluated_in_step(self, step):
    return step in self.evaluated_steps

  def add_evaluated_step(self, step):
    self.evaluated_steps.add(step)

  def log_in_step(self, step):
    return step in self.log_steps

  def add_log_step(self, step):
    self.log_steps.add(step)