import argparse
import os
import json
from collections import defaultdict


def convert_into_group(input_folder, output_folder):
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)
  for split in ["train", "dev", "test"]:

    with open(os.path.join(input_folder, "{}.json".format(split))) as fin:
      print("Processing {}".format(fin.name))
      data = json.load(fin)
      data_collection = defaultdict(list)
      for example in data:
        text = " ".join(example["token"])
        data_collection[text].append(example)
      print("There are {} examples, and {} sentences".format(
        len(data), len(data_collection)))
    with open(os.path.join(output_folder, "{}.json".format(split)), "w") as fout:
      print("Dumping results to {}".format(fout.name))
      for sentence, group in data_collection.items():
        tokens = sentence.split()
        new_example = {
          "sentence": sentence,
          "masked_sentence": None,
          "subj_index": [],
          "obj_index": [],
          "subj_type": [],
          "obj_type": [],
          "subj_text": [],
          "obj_text": [],
          "relation": []}
        entity_subj_spans = []
        entity_obj_spans = []
        relations = []
        for example in group:
          start_index = int(example["subj_start"])
          end_index = int(example["subj_end"]) + 1
          span = (start_index, end_index, example["subj_type"], " ".join(tokens[start_index: end_index]))
          entity_subj_spans.append(span)
          start_index = int(example["obj_start"])
          end_index = int(example["obj_end"]) + 1
          span = (start_index, end_index, example["obj_type"], " ".join(tokens[start_index: end_index]))
          entity_obj_spans.append(span)
          relations.append(example["relation"])
        entity_spans = sorted(list(set(entity_subj_spans + entity_obj_spans)), key=lambda x: x[0])
        new_sentence = []
        start_index = 0
        span_to_index = {}
        for span in entity_spans:
          new_sentence.extend(tokens[start_index: span[0]])
          span_to_index[span] = len(new_sentence)
          new_sentence.append("[{}]".format(span[2]))
          start_index = span[1]
        new_sentence.extend(tokens[start_index:])
        for span in entity_subj_spans:
          new_example["subj_index"].append(span_to_index[span])
          new_example["subj_type"].append(span[2])
          new_example["subj_text"].append(span[3])
        for span in entity_obj_spans:
          new_example["obj_index"].append(span_to_index[span])
          new_example["obj_type"].append(span[2])
          new_example["obj_text"].append(span[3])
        for relation in relations:
          new_example["relation"].append(relation)
        new_example["masked_sentence"] = new_sentence
        fout.write(json.dumps(new_example) + "\n")

def convert(input_folder, output_folder):
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)

  for split in ["train", "dev", "test"]:
    examples = []
    with open(os.path.join(input_folder, "{}.json".format(split))) as fin:
      print("Processing {}".format(fin.name))
      data = json.load(fin)
      for example in data:
        processed = {}
        tokens = example["token"]
        processed["text"] = " ".join(tokens)

        start_index = int(example["subj_start"])
        end_index = int(example["subj_end"]) + 1
        processed["subj_text"] = " ".join(tokens[start_index: end_index])
        processed["subj_span"] = (start_index, end_index)

        start_index = int(example["obj_start"])
        end_index = int(example["obj_end"]) + 1
        processed["obj_text"] = " ".join(tokens[start_index: end_index])
        processed["obj_span"] = (start_index, end_index)

        processed["subj_type"] = example["subj_type"]
        processed["obj_type"] = example["obj_type"]

        processed["label"] = example["relation"]
        examples.append(processed)
    with open(os.path.join(output_folder, "{}.json".format(split)), "w") as fout:
      for example in examples:
        fout.write(json.dumps(example) + "\n")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_folder", type=str)
  parser.add_argument("--output_folder", type=str)
  args = parser.parse_args()
  # convert_into_group(args.input_folder, args.output_folder)
  convert(args.input_folder, args.output_folder)