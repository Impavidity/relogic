import argparse
import json
from relogic.structures.document import Sentence
from tqdm import tqdm
from relogic.pipelines.core import Pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--data_file_path", type=str)
parser.add_argument("--output_file_path", type=str)

args = parser.parse_args()

pipeline = Pipeline(
  component_names=["predicate_detection", "srl"],
  component_model_names= {"predicate_detection" : "pd-conll12" ,"srl": "srl-conll12"})

fout = open(args.output_file_path, 'w')
with open(args.data_file_path) as fin:
  sentences = []
  for line in tqdm(fin):
    if line == "\n":
      pipeline.execute(sentences)
      for sent in sentences:
        fout.write(json.dumps(sent.convert_to_json()) + "\n")
      fout.write("\n")
    else:
      sentences.append(Sentence(text=line.strip("\n"), tokenizer="gpt2"))





