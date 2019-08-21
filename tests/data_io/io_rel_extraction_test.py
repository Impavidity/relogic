from collections import namedtuple
import torch

from relogic.logickit.dataset.minibatching import Minibatch
from relogic.logickit.tokenizer.tokenization import BertTokenizer
from relogic.logickit.data_io.io_relation import (
  get_relextraction_examples, convert_relextraction_examples_to_features,
  generate_rel_extraction_input)
from relogic.logickit.base import utils
from relogic.logickit.base.constants import NEVER_SPLIT

utils.log("Reading data")
examples = get_relextraction_examples(
  "data/raw_data/rel_extraction/tacred/origin/dev.json")
utils.log("Preprocessing...")
vocab_path = "relogic/logickit/vocabs/tacred-bert-base-cased-vocab.txt"
tokenizer = BertTokenizer.from_pretrained(
  vocab_path, do_lower_case=True, never_split=NEVER_SPLIT, lang="en")
extra_args = {
  "entity_surface_aware": False,
  "label_mapping":
  utils.load_pickle("data/preprocessed_data/rel_extraction_tacred_label_mapping.pkl")
}
# with Pool(10) as p:
#   p.map(lambda e: e.process(tokenizer, extra_args), examples)
for example in examples:
  example.process(tokenizer, extra_args)
utils.log("Done")
example_batch = examples[:5]
feature_batch = convert_relextraction_examples_to_features(example_batch, 512)
batch = Minibatch(
  task_name="rel_extraction",
  size=5,
  examples=example_batch,
  ids=None,
  teacher_predictions=None,
  input_features=feature_batch)
Config = namedtuple('Config', ['use_dependency_feature'])
config = Config(use_dependency_feature=False)
for example in batch.examples:
  print(example.tokens)
input = generate_rel_extraction_input(batch, config, torch.device("cuda:0"),
                                      True)
print(input)
