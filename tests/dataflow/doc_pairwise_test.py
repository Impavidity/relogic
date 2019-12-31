import json
from types import SimpleNamespace

import relogic.utils.crash_on_ipy
from relogic.logickit.base.constants import GCN_DOC
from relogic.logickit.dataflow import TASK_TO_DATAFLOW_CLASS_MAP, DocPairwiseDataFlow
from relogic.logickit.tokenizer.tokenization import BertTokenizer

config = SimpleNamespace(
  **{
    "max_seq_length": 450,
    "regression": False,
    "label_mapping_path": "data/preprocessed_data/binary_classification.json",
    "tasks": {
      GCN_DOC: {
        "selected_non_final_layers": None
      }
    }
  })

tokenizers = {
  "BPE": BertTokenizer.from_pretrained("bert-large-uncased"),
}

dataflow: DocPairwiseDataFlow = TASK_TO_DATAFLOW_CLASS_MAP[GCN_DOC](
  task_name=GCN_DOC,
  config=config,
  tokenizers=tokenizers,
  label_mapping=json.load(open(config.label_mapping_path)))

dataflow.update_with_file("tests/datasets/robust04/train.json")

for mb in dataflow.get_minibatches(minibatch_size=2, sequential=False):
  print(mb)
  raise NotImplementedError("You can start to play with data")
