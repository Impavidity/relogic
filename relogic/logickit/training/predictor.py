from relogic.logickit.tokenizer.tokenization import BertTokenizer
from relogic.logickit.tasks import get_task
from relogic.logickit.tasks.task import Task
from relogic.logickit.model import get_model
from relogic.logickit.dataflow import DataFlow
import torch
from relogic.logickit.base import utils
from relogic.logickit.base.constants import IR_TASK, ECP_TASK

class Predictor(object):
  def __init__(self, config):
    self.config = config
    self.tokenizer = {
      "BPE": BertTokenizer.from_pretrained(
        config.vocab_path, do_lower_case=config.do_lower_case,
        never_split=config.never_split, lang=config.lang),
      # "Fasttext": FasttextTokenizer.from_pretrained("wiki-news-300d-1M")
    }
    self.tasks = [
      get_task(self.config, task_name, self.tokenizer)
      for task_name in self.config.task_names
    ]
    # self.tokenizer = BertTokenizer.from_pretrained(
    #   config.vocab_path, do_lower_case=config.do_lower_case,
    #   never_split=config.never_split, lang=config.lang)
    # self.tasks = [
    #   get_task(self.config, task_name, self.tokenizer)
    #   for task_name in self.config.task_names
    # ]
    self.model = get_model(config)(config=self.config, tasks=self.tasks)

  def predict(self, structures, selected_task=None):
    # We currently assume it is single task model
    sel_task = None
    if selected_task is not None:
      for task in self.tasks:
        if task.name == selected_task:
          sel_task = task
    task: Task = self.tasks[0] if selected_task is None else sel_task
    data: DataFlow = task.dataset
    data.update_with_structures(structures)
    print(self.config)
    print("Updated the structures")
    # results = []
    for i, mb in enumerate(data.get_minibatches(self.config.tasks[task.name]["test_batch_size"])):
      # batch_preds = self.model.test(mb)
      batch_preds = self.model.test_abstract(mb)
      batch_preds = batch_preds[task.name]["logits"]
      labels = data.decode_to_labels(batch_preds, mb)
      yield batch_preds, labels
      # results.append(batch_preds.data.cpu().numpy())
      # labels = data.decode_to_labels(batch_preds)
    #   if i % 100 == 0:
    #     print("Processed {} batches".format(i))
    # return results

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


