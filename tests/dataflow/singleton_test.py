import json
from types import SimpleNamespace

import relogic.utils.crash_on_ipy
from relogic.logickit.base.constants import SEQUENCE_CLASSIFICATION_TASK
from relogic.logickit.dataflow import TASK_TO_DATAFLOW_CLASS_MAP, SingletonDataFlow
from relogic.logickit.tokenizer.tokenization import BertTokenizer

config = SimpleNamespace(
  **{
    "buckets": [(0, 15), (15, 40), (40, 450)],
    "max_seq_length": 450,
    "label_mapping_path": "data/preprocessed_data/entity_type_classification.json"
  })

tokenizers = {
  "BPE": BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case=False),
}

dataflow: SingletonDataFlow = TASK_TO_DATAFLOW_CLASS_MAP[SEQUENCE_CLASSIFICATION_TASK](
  task_name=SEQUENCE_CLASSIFICATION_TASK,
  config=config,
  tokenizers=tokenizers,
  label_mapping=json.load(open(config.label_mapping_path)))

examples = [{
  "tokens": ["I", "visited", "China", "yesterday", "."],
  "labels": ["LOC"]
},{
  "tokens": ["Barack", "Obama", "went", "to", "Paris", "."],
  "labels": ["PER", "LOC"]
}]

dataflow.update_with_jsons(examples)

for mb in dataflow.get_minibatches(minibatch_size=2):
  print(mb)

raise NotImplementedError("You can start to play with data")