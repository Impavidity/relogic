import torch
from relogic.logickit.base import utils

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