import json
from types import SimpleNamespace

import relogic.utils.crash_on_ipy
from relogic.logickit.base.constants import DEP_PARSING_TASK
from relogic.logickit.dataflow import TASK_TO_DATAFLOW_CLASS_MAP, DependencyParsingDataFlow
from relogic.logickit.tokenizer.tokenization import BertTokenizer

config = SimpleNamespace(
  **{
    "buckets": [(0, 15), (15, 40), (40, 450)],
    "max_seq_length": 450,
    "label_mapping_path": "data/preprocessed_data/universal_dependency_labels.json"
  })

tokenizers = {
  "BPE": BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case=False, pretokenized=True),
}
dataflow: DependencyParsingDataFlow = TASK_TO_DATAFLOW_CLASS_MAP[DEP_PARSING_TASK](
  task_name=DEP_PARSING_TASK,
  config=config,
  tokenizers=tokenizers,
  label_mapping=json.load(open(config.label_mapping_path)))

examples = [
{"tokens": ["Al", "-", "Zaman", ":", "American", "forces", "killed", "Shaikh", "Abdullah", "al", "-", "Ani", ",",
            "the", "preacher", "at", "the", "mosque", "in", "the", "town", "of", "Qaim", ",", "near", "the", "Syrian",
            "border", "."],
 "labels": ["root", "punct", "flat", "punct", "amod", "nsubj", "parataxis", "obj", "flat", "flat", "punct", "flat",
            "punct", "det", "appos", "case", "det", "obl", "case", "det", "nmod", "case", "nmod", "punct", "case",
            "det", "amod", "nmod", "punct"],
 "arcs": [0, 1, 1, 1, 6, 7, 1, 7, 8, 8, 8, 8, 8, 15, 8, 18, 18, 7, 21, 21, 18, 23, 21, 21, 28, 28, 28, 21, 1],
 "pos": ["PROPN", "PUNCT", "PROPN", "PUNCT", "ADJ", "NOUN", "VERB", "PROPN", "PROPN", "PROPN", "PUNCT", "PROPN",
         "PUNCT", "DET", "NOUN", "ADP", "DET", "NOUN", "ADP", "DET", "NOUN", "ADP", "PROPN", "PUNCT", "ADP", "DET",
         "ADJ", "NOUN", "PUNCT"]},
{"tokens": ["[", "This", "killing", "of", "a", "respected", "cleric", "will", "be", "causing", "us", "trouble", "for",
            "years", "to", "come", ".", "]"],
 "labels": ["punct", "det", "nsubj", "case", "det", "amod", "nmod", "aux", "aux", "root", "iobj", "obj", "case",
            "obl", "mark", "acl", "punct", "punct"],
 "arcs": [10, 3, 10, 7, 7, 7, 3, 10, 10, 0, 10, 10, 14, 10, 16, 14, 10, 10],
 "pos": ["PUNCT", "DET", "NOUN", "ADP", "DET", "ADJ", "NOUN", "AUX", "AUX", "VERB", "PRON", "NOUN", "ADP", "NOUN",
         "PART", "VERB", "PUNCT", "PUNCT"]}
]

dataflow.update_with_jsons(examples)

for mb in dataflow.get_minibatches(minibatch_size=2):
  print(mb)

raise NotImplementedError("You can start to play with data")