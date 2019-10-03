import json
from types import SimpleNamespace

import relogic.utils.crash_on_ipy
from relogic.logickit.base.constants import PARALLEL_MAPPING_TASK
from relogic.logickit.dataflow import TASK_TO_DATAFLOW_CLASS_MAP, ParallelDataFlow
from relogic.logickit.tokenizer.tokenization import BertTokenizer

config = SimpleNamespace(
  **{
    "buckets": [(0, 100), (100, 250), (250, 512)],
    "max_seq_length": 512
  })

tokenizers = {
  "BPE": BertTokenizer.from_pretrained("bert-base-multilingual-cased"),
}

dataflow: ParallelDataFlow = TASK_TO_DATAFLOW_CLASS_MAP[PARALLEL_MAPPING_TASK](
  task_name=PARALLEL_MAPPING_TASK,
  config=config,
  tokenizers=tokenizers,
  label_mapping=None)

examples = [{
  "text_a": "Die deutschsprachige Gemeinschaft in Belgien",
  "text_b": "Germanspeaking Community of Belgium",
  "alignment": [[0, 2, 3, 4], [0, 1, 2, 3]]
}, {
  "text_a": "The European Social Fund , created in 1957 , is the European Union ’ s main financial instrument for investing in people .",
  "text_b": "Het Europees Sociaal Fonds ( ESF ) , dat werd opgericht in 1957 , is het voornaamste nanciële instrument van de Europese Unie om te investeren in mensen .",
  "alignment": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18, 19, 20, 21, 22],
                [0, 1, 2, 3, 7, 10, 11, 12, 13, 14, 15, 21, 22, 17, 16, 18, 23, 24, 26, 27, 28]]
}]

dataflow.update_with_jsons(examples)

for mb in dataflow.get_minibatches(minibatch_size=2):
  print(mb)

raise NotImplementedError("You can start to play with data")
