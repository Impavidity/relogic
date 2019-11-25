from relogic.logickit.base.constants import AUXILIARY_TRAINING, ITERATIVE_TRAINING
import numpy as np
import bisect
import json

def auxiliary_training(config, tasks):
  training_scheme = json.load(open(config.training_scheme_file))
  datasets = {task.name: task.train_set for task in tasks}
  mbs = {task.name: task.train_set.endless_minibatches(config.tasks[task.name]["train_batch_size"],
                                                       sequential=(config.tasks[task.name]["dataset_type"] == "sequential"))
          for task in tasks}
  warmup = training_scheme["warmup"]

  warmup_tasks = warmup["tasks"]
  warmup_task_weights = [np.sqrt(datasets[task_name].size) for task_name in warmup_tasks]
  warmup_id2task = {idx: task for idx, task in enumerate(warmup_tasks)}
  warmup_thresholds = np.cumsum([w / np.sum(warmup_task_weights) for w in warmup_task_weights])
  warmup_steps = warmup["steps"]
  # raise NotImplementedError()
  for i in range(warmup_steps):
    dataset_ind = bisect.bisect(warmup_thresholds, np.random.random())
    yield next(mbs[warmup_id2task[dataset_ind]])

  auxiliary = training_scheme["auxiliary"]
  auxiliary_tasks = auxiliary["tasks"]
  while True:
    for auxiliary_task in auxiliary_tasks:
      dataset_ind = bisect.bisect(warmup_thresholds, np.random.random())
      yield next(mbs[warmup_id2task[dataset_ind]])
      yield next(mbs[auxiliary_task])

def iterative_training(config, tasks):
  training_scheme = json.load(open(config.training_scheme_file))
  mbs = {task.name: task.train_set.endless_minibatches(config.tasks[task.name]["train_batch_size"],
                                                       sequential=(config.tasks[task.name]["dataset_type"] == "sequential"))
         for task in tasks}

  iterative_stages = training_scheme["iterative_stages"]
  while True:
    for stage in iterative_stages:
      stage_steps = stage["steps"]
      stage_tasks = stage["tasks"]
      for i in range(stage_steps):
        for task in stage_tasks:
          yield next(mbs[task])

TRAINING_SCHEME = {
  AUXILIARY_TRAINING: auxiliary_training,
  ITERATIVE_TRAINING: iterative_training,
}