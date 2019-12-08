import json
from types import SimpleNamespace

import relogic.utils.crash_on_ipy
from relogic.logickit.base.constants import DOCIR_TASK
from relogic.logickit.dataflow import TASK_TO_DATAFLOW_CLASS_MAP, DocPointwiseDataFlow
from relogic.logickit.tokenizer.tokenization import BertTokenizer

config = SimpleNamespace(
  **{
    "buckets": [(0, 15)],
    "max_seq_length": 450,
    "label_mapping_path": "data/preprocessed_data/binary_classification.json"
  })

tokenizers = {
  "BPE": BertTokenizer.from_pretrained("bert-base-multilingual-cased"),
}

dataflow: DocPointwiseDataFlow = TASK_TO_DATAFLOW_CLASS_MAP[DOCIR_TASK](
  task_name=DOCIR_TASK,
  config=config,
  tokenizers=tokenizers,
  label_mapping=json.load(open(config.label_mapping_path)))

examples = [{
  "text_a": "bbc world service staff cuts",
  "text_b":
  ["gossip day by day : bbc world service to cut five language services",
   "stoke city : begovic wilkinson shawcross wilson wilkinson walters whelan nzonzi kightly jerome crouch"],
  "label": "1"
}, {
  "text_a": "barbara walters chicken pox",
  "text_b": [
  "gossip day by day : bbc world service to cut five language services",
  "stoke city : begovic wilkinson shawcross wilson wilkinson walters whelan nzonzi kightly jerome crouch"],
  "label": "0"
}]

dataflow.update_with_jsons(examples)

for mb in dataflow.get_minibatches(minibatch_size=1):
  print(mb)

raise NotImplementedError("You can start to play with data")
