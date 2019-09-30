from relogic.logickit.tasks.task import Task
from relogic.logickit.dataset.labeled_data_loader import LabeledDataLoader
from relogic.logickit.scorer.span_extraction_scorer import SpanExtractionScorer
from relogic.logickit.modules.span_extraction_module import SpanExtractionModule
from relogic.logickit.modules.span_gcn import SpanGCNModule


class SpanExtraction(Task):
  def __init__(self, config, name, tokenizer=None):
    super(SpanExtraction, self).__init__(
      config, name, LabeledDataLoader(config, name, tokenizer))
    self.n_classes = 2
    self.config = config

  def get_module(self):
    if self.name in ["squad11", "squad20"]:
      return SpanExtractionModule(self.config, self.name, self.n_classes)
    else:
      raise ValueError("Can not find task name {}".format(self.name))


  def get_scorer(self, dump_to_file=None):
    if self.name in ["squad11", "squad20"]:
      return SpanExtractionScorer(
        dataset=self.name,
        gold_answer_file=self.config.gold_answer_file,
        null_score_diff_threshold=self.config.null_score_diff_threshold,
        dump_to_file=dump_to_file)

class ECPExtraction(Task):
  def __init__(self, config, name, tokenizer=None):
    super(ECPExtraction, self).__init__(
      config, name, LabeledDataLoader(config, name, tokenizer))
    self.n_classes = len(set(self.loader.label_mapping.values()))
    self.config = config




class SpanGCN(Task):
  def __init__(self, config, name, tokenizer=None):
    super(SpanGCN, self).__init__(
      config, name, LabeledDataLoader(config, name, tokenizer))
    self.span_n_classes = len(set(self.loader.label_mapping.values()))

  def get_module(self):
    span_n_classes = len(self.loader.label_mapping["span"])
    label_n_classes = len(self.loader.label_mapping["label"])
    return SpanGCNModule(
      config=self.config,
      task_name=self.name,
      span_n_classes=span_n_classes,
      label_n_classes=label_n_classes,)

  def get_scorer(self, dump_to_file=None):
    pass