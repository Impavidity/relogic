from collections import namedtuple
import torch
from relogic.logickit.dataset.minibatching import Minibatch
from relogic.logickit.tokenizer.tokenization import BertTokenizer
from relogic.logickit.data_io.io_srl import (
  get_srl_examples, convert_srl_examples_to_features, generate_srl_input)
from relogic.logickit.base import utils


utils.log("Reading data")
examples = get_srl_examples(
  "data/raw_data/srl/json/conll05/propbank/span_sample.json")
utils.log("Preprocessing")
tokenizer = BertTokenizer.from_pretrained(
  "bert-base-cased",
  do_lower_case=False)
extra_args = {
  "label_mapping": utils.load_pickle("data/preprocessed_data/srl_conll05_label_mapping.pkl"),
  "predicate_surface_aware": True,
  "srl_module_type": "span_gcn"
}

for example in examples:
  example.process(tokenizer, extra_args)
utils.log("Done")
from random import sample

example_batch = sample(examples, 2)
feature_batch = convert_srl_examples_to_features(example_batch, max_seq_length=512)
# for feature in feature_batch:
#   print(feature.__dict__)
batch = Minibatch(
  task_name="srl_conll05",
  size=5,
  examples=example_batch,
  ids=None,
  teacher_predictions=None,
  input_features=feature_batch)
Config = namedtuple('Config', ['use_span_candidates', "srl_module_type", "use_description"])
config = Config(use_span_candidates=True, srl_module_type="span_gcn", use_description=True)
input = generate_srl_input(batch, config, torch.device("cuda:6"), True)
print(input["extra_args"]["predicate_descriptions_ids"].size())
print(input["extra_args"]["argument_descriptions_ids"].size())
print(input["extra_args"]["argument_descriptions_ids"])