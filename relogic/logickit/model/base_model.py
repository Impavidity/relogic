import torch
from relogic.logickit.base import utils
from torch.utils.tensorboard import SummaryWriter

class BaseModel(object):
  """
  Basic components for models
  """
  def __init__(self, config, ext_config):
    self.config = config
    self.ext_config = ext_config
    if config.local_rank == -1 or config.no_cuda:
      self.device = torch.device("cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu")
      n_gpu = torch.cuda.device_count()
    else:
      torch.cuda.set_device(config.local_rank)
      self.device = torch.device("cuda:" + str(config.local_rank))
      n_gpu = 1
      # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
      # torch.distributed.init_process_group(backend='nccl')
    utils.log("device: {}".format(self.device))
    self.tb_writer = SummaryWriter(log_dir=self.config.tb_writer_log_dir)



  # def train_labeled_abstract(self, mb, step):
  #   self.model

  def setup_training(self, config, tasks):
    size_train_examples = 0
    config.num_steps_in_one_epoch = 0
    if config.mode == "train" or config.mode == "finetune":
      for task in tasks:
        utils.log("{} : {}  training examples".format(task.name, task.train_set.size))
        if "loss_weight" in config.tasks[task.name]:
          utils.log("loss weight {}".format(config.tasks[task.name]["loss_weight"]))
        size_train_examples += task.train_set.size
        config.num_steps_in_one_epoch += task.train_set.size // config.tasks[task.name]["train_batch_size"]

        config.tasks[task.name]["train_batch_size"] = config.tasks[task.name][
                                                        "train_batch_size"] // config.gradient_accumulation_steps
        config.tasks[task.name]["test_batch_size"] = config.tasks[task.name][
                                                       "test_batch_size"] // config.gradient_accumulation_steps
        utils.log("Training batch size: {}".format(config.tasks[task.name]["train_batch_size"]))
    calculated_num_train_optimization_steps = config.num_steps_in_one_epoch * config.epoch_number \
      if config.schedule_lr else -1
    if config.num_train_optimization_steps == 0:
      config.num_train_optimization_steps = calculated_num_train_optimization_steps
    else:
      utils.log("Overwriting the training steps to {} instead of {} because of the configuration".format(
        config.num_train_optimization_steps, calculated_num_train_optimization_steps))
    utils.log("Optimization steps : {}".format(config.num_train_optimization_steps))