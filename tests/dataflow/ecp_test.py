from relogic.logickit.dataflow import TASK_TO_DATAFLOW_CLASS_MAP, ECPDataFlow
from relogic.logickit.base.constants import ECP_TASK
from types import SimpleNamespace
import json
from relogic.logickit.tokenizer.tokenization import BertTokenizer
import relogic.utils.crash_on_ipy

config = SimpleNamespace(**{
  "buckets": [(0, 90), (90, 150), (150, 200), (200, 450)],
  "max_seq_length": 450,
  "label_mapping_path": "data/preprocessed_data/binary_classification.json"})

tokenizers = {
  "BPE": BertTokenizer.from_pretrained("bert-base-multilingual-cased", lang="zh")
}

dataflow: ECPDataFlow = TASK_TO_DATAFLOW_CLASS_MAP[ECP_TASK](
  task_name=ECP_TASK,
  config=config,
  tokenizers=tokenizers,
  label_mapping=json.load(open(config.label_mapping_path)))


examples = [{
  "tokens": ['当', '我', '看', '到', '建', '议', '被', '采', '纳', ',', '部', '委', '领', '导', '写', '给', '我', '的', '回', '信', '时', ',', '我', '知', '道', '我', '正', '在', '为', '这', '个', '国', '家', '的', '发', '展', '尽', '着', '一', '份', '力', '量', ',', '2', '7', '日', ',', '河', '北', '省', '邢', '台', '钢', '铁', '有', '限', '公', '司', '的', '普', '通', '工', '人', '白', '金', '跃', ',', '拿', '着', '历', '年', '来', '国', '家', '各', '部', '委', '反', '馈', '给', '他', '的', '感', '谢', '信', ',', '激', '动', '地', '对', '中', '新', '网', '记', '者', '说', ',', '2', '7', '年', '来', ',', '国', '家', '公', '安', '部', '国', '家', '工', '商', '总', '局', '国', '家', '科', '学', '技', '术', '委', '员', '会', '科', '技', '部', '卫', '生', '部', '国', '家', '发', '展', '改', '革', '委', '员', '会', '等', '部', '委', '均', '接', '受', '并', '采', '纳', '过', '的', '我', '的', '建', '议'],
  "labels": [(86, 96, 102, 152)],
  "clause_spans": [(0, 9), (10, 21), (22, 42), (43, 46), (47, 66), (67, 85), (86, 96), (97, 101), (102, 152)]
}]

# Some problem with Chinese Indexing
# for token list (Chinese characters), they will be combined into a string without space.
# Until now the spans index is still correct.
# But after tokenization, something goes wrong. Because some numbers will be merged (several digits will be merged)
# Then the index information is wrong here.


dataflow.update_with_jsons(examples)

for mb in dataflow.get_minibatches(minibatch_size=1):
  print(mb)

raise NotImplementedError("You can start to play with data")


