from collections import namedtuple
import torch
from relogic.logickit.dataset.minibatching import Minibatch
from relogic.logickit.tokenizer.tokenization import BertTokenizer
from relogic.logickit.data_io.io_srl import (
  get_srl_examples, convert_srl_examples_to_features, generate_srl_input)
from relogic.logickit.base import utils


utils.log("Reading data")
examples = get_srl_examples(
  "data/raw_data/srl/json/conll05/boundary/test.json")
utils.log("Preprocessing")
tokenizer = BertTokenizer.from_pretrained(
  "bert-base-cased",
  do_lower_case=False)
extra_args = {
  "label_mapping": utils.load_pickle("data/preprocessed_data/BIO_label_mapping.pkl"),
  "predicate_surface_aware": True
}

for example in examples:
  example.process(tokenizer, extra_args)
utils.log("Done")
from random import sample
example_batch = sample(examples, 5)
feature_batch = convert_srl_examples_to_features(example_batch, max_seq_length=512)
batch = Minibatch(
  task_name="srl_conll05",
  size=5,
  examples=example_batch,
  ids=None,
  teacher_predictions=None,
  input_features=feature_batch)

input = generate_srl_input(batch, None, torch.device("cuda:6"), True)
for example in example_batch:
  print(example.text_tokens)
print(input)