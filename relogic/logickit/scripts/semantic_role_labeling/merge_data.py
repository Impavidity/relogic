from relogic.logickit.utils.utils import get_span_labels
import argparse
import json

def merge(gold_examples, predicted_examples):
  assert len(gold_examples) == len(predicted_examples)
  for gold_example, predicted_example in zip(gold_examples, predicted_examples):
    predicted_spans, labels = get_span_labels(predicted_example["label"])
    predicted_spans = [(span[0], span[1]) for span in sorted(predicted_spans, key=lambda span: span[0], reverse=False)]
    gold_example["span_candidates"] = predicted_spans


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--gold_data_path", type=str)
  parser.add_argument("--pred_data_path", type=str)
  parser.add_argument("--output_data_path", type=str)
  args = parser.parse_args()
  with open(args.gold_data_path) as fin:
    gold_examples = []
    for line in fin:
      gold_examples.append(json.loads(line))
  with open(args.pred_data_path) as fin:
    predicted_examples = []
    for line in fin:
      predicted_examples.append(json.loads(line))
  merge(gold_examples, predicted_examples)
  with open(args.output_data_path, "w") as fout:
    for example in gold_examples:
      fout.write(json.dumps(example) + "\n")
  

