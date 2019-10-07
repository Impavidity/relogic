from typing import List
from types import SimpleNamespace

import torch

import relogic.utils.crash_on_ipy
from relogic.logickit.base.constants import PAIRWISE_TASK
from relogic.logickit.dataflow import TASK_TO_DATAFLOW_CLASS_MAP, PairwiseDataFlow
from relogic.logickit.tokenizer.tokenization import BertTokenizer

config = SimpleNamespace(
  **{
    "buckets": [(0, 15), (15, 40), (40, 450)],
    "max_seq_length": 450
  }
)

tokenizers = {
  "BPE": BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case=True, lang="zh"),
}

dataflow: PairwiseDataFlow = TASK_TO_DATAFLOW_CLASS_MAP[PAIRWISE_TASK](
  task_name=PAIRWISE_TASK,
  config=config,
  tokenizers=tokenizers,
  label_mapping=None)

examples = [{
  "guid": "1",
  "text": "林郑月娥，香港特別行政区現任行政長官，在36年間曾於20個政府不同的工作崗位上工作。",
  "text_p": "Carrie Lam Cheng Yuet-ngor is a Hong Kong politician serving as the 4th Chief Executive of Hong Kong since 2017.",
  "text_n": "Leung Chun-ying, also known as CY Leung, is a Hong Kong politician. He served as the third Chief Executive of Hong Kong between 2012 and 2017."
}, {
  "guid": "2",
  "text": "天安门广场是位于北京市中心的城市广场，是世界上最大的城市广场之一。因位于明清北京皇城的南门「天安门」外而得名。广场北端设有国旗杆，每天都會隨日出、日落進行升旗、降旗仪式。",
  "text_p": "Tiananmen Square or Tian'anmen Square is a city square in the centre of Beijing, China, named after the Tiananmen located to its north, separating it from the Forbidden City.",
  "text_n": "The Forbidden City is a palace complex in central Beijing, China. It houses the Palace Museum, and was the former Chinese imperial palace from the Ming dynasty to the end of the Qing dynasty (the years 1420 to 1912)."
}]

dataflow.update_with_jsons(examples)

for mb in dataflow.get_minibatches(minibatch_size=2):
  print(mb)

raise NotImplementedError("You can start to play with data")