from collections import namedtuple
from dataset.minibatching import Minibatch
from multiprocessing import Pool
import torch
from tokenizer.tokenization import BertTokenizer

from data_io.io_rel_extraction import (
  get_relextraction_examples,
  convert_relextraction_examples_to_features,
  generate_rel_extraction_input)


from base import utils
from base.constants import NEVER_SPLIT
utils.log("Reading data")
examples = get_relextraction_examples("data/raw_data/rel_extraction/tacred/dev_dep.json")
utils.log("Preprocessing...")
vocab_path = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(
    vocab_path, do_lower_case=True,
    never_split=NEVER_SPLIT, lang="en")
extra_args = {
  "entity_surface_aware": False,
  "label_mapping": utils.load_pickle("data/preprocessed_data/rel_extraction_tacred_label_mapping.pkl")
}
# with Pool(10) as p:
#   p.map(lambda e: e.process(tokenizer, extra_args), examples)
for example in examples:
  example.process(tokenizer, extra_args)
utils.log("Done")
example_batch = examples[:5]
feature_batch = convert_relextraction_examples_to_features(example_batch, 512)
batch = Minibatch(
  task_name="rel_extraction", size=5, examples=example_batch,
  ids=None, teacher_predictions=None, input_features=feature_batch)
Config = namedtuple('Config', ['use_dependency_feature'])
config = Config(use_dependency_feature=True)
input = generate_rel_extraction_input(batch, config, torch.device("cuda:6"), True)
print(input)
