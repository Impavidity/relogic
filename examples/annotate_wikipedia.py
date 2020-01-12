import argparse
import json
from relogic.structures.document import Document
from tqdm import tqdm
from relogic.pipelines.core import Pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--data_file_path", type=str)
parser.add_argument("--output_file_path", type=str)

args = parser.parse_args()

pipeline = Pipeline(
  component_names=["predicate_detection", "srl"],
  component_model_names= {"predicate_detection" : "spacy" ,"srl": "srl-conll12"})

fout = open(args.output_file_path, 'w')
with open(args.data_file_path) as fin:
  for line in tqdm(fin):
    example = json.loads(line)
    document = Document(text=example["text"])
    sentences = []
    for para in document.paragraphs:
      for sent in para.sentences:
        sentences.append(sent)
    pipeline.execute(sentences)
    fout.write(json.dumps({
      "id": example["id"],
      "url": example["url"],
      "title": example["title"],
      "sents": [sent.convert_to_json() for sent in sentences]
    }) + "\n")




