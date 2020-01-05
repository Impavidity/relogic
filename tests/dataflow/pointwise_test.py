import json
from types import SimpleNamespace

import relogic.utils.crash_on_ipy
from relogic.logickit.base.constants import POINTWISE_TASK
from relogic.logickit.dataflow import TASK_TO_DATAFLOW_CLASS_MAP, PointwiseDataFlow
from relogic.logickit.tokenizer.tokenization import BertTokenizer

config = SimpleNamespace(
  **{
    "buckets": [(0, 15), (15, 40), (40, 450)],
    "max_seq_length": 450,
    "label_mapping_path": "data/preprocessed_data/binary_classification.json",
    "regression": False,
    "doc_ir_model": "evidence",
    "tasks": {
      POINTWISE_TASK: {"selected_non_final_layers": 8}
    }
  })

tokenizers = {
  "BPE": BertTokenizer.from_pretrained("bert-base-multilingual-cased"),
}

dataflow: PointwiseDataFlow = TASK_TO_DATAFLOW_CLASS_MAP[POINTWISE_TASK](
  task_name=POINTWISE_TASK,
  config=config,
  tokenizers=tokenizers,
  label_mapping=json.load(open(config.label_mapping_path)))

examples = [{
  "text_a": "bbc world service staff cuts",
  "text_b":
  "gossip day by day : bbc world service to cut five language services",
  "selected_a_indices": [0, 1, 2, 3, 4],
  "sequence_labels": ["I", "I", "I", "I", "O", "O", "O", "O", "O", "O", "O", "O", "O"],
  "label": "1"
}, {
  "text_a": "barbara walters chicken pox",
  "text_b":
  "stoke city : begovic wilkinson shawcross wilson wilkinson walters whelan nzonzi kightly jerome crouch",
  "selected_a_indices": [0, 1, 2, 3],
  "label": "0"
}]

dataflow.update_with_jsons(examples)

for mb in dataflow.get_minibatches(minibatch_size=2):
  print(mb)

raise NotImplementedError("You can start to play with data")
