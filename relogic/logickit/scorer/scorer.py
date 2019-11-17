import abc
import torch

class Scorer(object, metaclass=abc.ABCMeta):
  def __init__(self):
    self._updated = False
    self._cached_results = {}
    self.results = None
    self.need_to_clear_output = False
    self.dump_to_file_path = None
    self.dump_to_file_handler = None


  @abc.abstractmethod
  def update(self, mbs, predictions, loss, extra_args):
    self._updated = True


  @abc.abstractmethod
  def get_loss(self):
    pass

  @abc.abstractmethod
  def _get_results(self):
    return []

  def get_results(self):
    if self.dump_to_file_path:
      self.need_to_clear_output = True
    results = self._get_results()
    self.results = results
    return results

  def results_str(self):
    return " - ".join(["{:}: {:.2f}".format(k, v)
                       for k, v in self.results])

  @staticmethod
  def unwrap_to_tensors(*tensors: torch.Tensor):
    """
    If you actually passed gradient-tracking Tensors to a Metric, there will be
    a huge memory leak, because it will prevent garbage collection for the computation
    graph. This method ensures that you're using tensors directly and that they are on
    the CPU.
    """
    return (x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in tensors)