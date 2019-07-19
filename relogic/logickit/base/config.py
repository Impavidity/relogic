from dataclasses import dataclass

@dataclass
class GeneralConfig(object):
  """Model configuration for training and test.
  
  Args:
    mode (str): The trainer mode can be one of the following choice:
      "train", "dev", "eval", "finetune"
    output_dir (str): A full directory path for saving the model configuration,
      parameters and prediction dumps.
    restore_path (str): Model configuration and parameters will be 
      restored from this full directory path.
    max_seq_length (int): Limited by the contextual model and GPU memory,
      the max sentence length is limited in this model. Default: 450.
    max_query_length (int): This is reserved for reading comprehension model.
      There are two components in reading comprehension model: query and paragraph.
      In the model, the query max length is limited by this argument. Default: 64.
    doc_stride (int): This is reserved for reading comprehension model. For long paragraph,
      it will be devided into several sub-paragraph by a sliding window whose window size
      is determined by the this argument. Default: 128.
    do_lower_case (bool): If True, convert the original text to lowercase. Default: False
    train_file (str): The file names of train file in :data:`raw_data_path`, segmented by ','. 
    dev_file (str): The file names of dev file in :data:`raw_data_path`, segmented by ','.
    test_file (str): The file names of test file in :data:`raw_data_path`, segmented by ','.
    
  """
  # IO
  mode: str
  output_dir: str
  restore_path: str

  max_seq_length: int = 450
  max_query_length: int = 64
  doc_stride: int = 128

  do_lower_case: bool = False

  train_file: str = "train.json"
  dev_file: str = "dev.json"
  test_file: str = "test.json"

  
  # Task Definition
  
  # Task Related Configuration
  
  # Hyper-parameter
  
  # Training
  
  # Semi-supervised
  
  # Training
  
  def load_from_json(self, config):
    pass
  
  def load_from_namespace(self, config):
    pass
  
  def load_from_json_file(self, config_path):
    pass
  