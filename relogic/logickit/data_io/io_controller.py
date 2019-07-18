import argparse

class IOController(object):
  """IO Controller class.
  IO controller is the main component of a `Dataset`
  This class is used to coordinate the following functions:
    - Get labeled examples from files and save as `Example`
    - Dump processed data to HDF5
    - Padding each feature in `Example`, and generate `Feature`
    - Batching the features and generate input for different tasks

  Args:
    config (argparse.Namespace):
  """
  def __init__(self, config: argparse.Namespace) -> None:
    self.config = config

