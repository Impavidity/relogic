import argparse
import json
import copy

from relogic.logickit.utils import get_span_labels

def ner_analysis(file_path):
  """
  We assume the file is in the following format
  {"tokens": List[str],
   "labels": List[str],
   "predicted_labels": List[str]}

  We categorize the error types into four types:
    type-error: correct entity boundary but wrong entity type
    boundary-error: wrong boundary
    extra-entity-error: predicting non-entity phrase as extra entity
    miss-entity-error: missing an entity
  """
  examples = []
  with open(file_path) as fin:
    for line in fin:
      examples.append(json.loads(line))
  type_error = 0
  boundary_error = 0
  extra_entity_error = 0
  miss_entity_error = 0
  for example in examples:
    gold_entities, _ = get_span_labels(example["labels"])
    pred_entities, _ = get_span_labels(example["predicted_labels"])
    if len(gold_entities - pred_entities) == 0:
      continue
    else:
      error_summary = {}
      error_summary["tokens"] = " ".join(example["tokens"])
      error_summary["type-error"] = []
      error_summary["boundary-error"] = []
      error_summary["extra-entity-error"] = []
      error_summary["miss-entity-error"] = []
      potential_miss_entity = gold_entities - pred_entities
      potential_extra_entity = pred_entities - gold_entities
      copy_potential_extra_entity = copy.deepcopy(potential_extra_entity)
      for span in potential_extra_entity:
        for gold_entity in gold_entities:
          if span[0] == gold_entity[0] and span[1] == gold_entity[1]:
            # This is type-error
            if span in copy_potential_extra_entity:
              error_summary["type-error"].append({
                "gold_entity_str": " ".join(example["tokens"][gold_entity[0]: gold_entity[1]+1]),
                "gold_entity_span": gold_entity,
                "pred_entity_str": " ".join(example["tokens"][span[0]: span[1]+1]),
                "pred_entity_span": span})
              type_error += 1
              copy_potential_extra_entity.remove(span)
              if gold_entity in potential_miss_entity:
                potential_miss_entity.remove(gold_entity)
          elif (span[0] >= gold_entity[0] and span[0] <= gold_entity[1]) \
                or (span[1] >= gold_entity[0] and span[1] <= gold_entity[1]) \
                or (span[0] <= gold_entity[0] and span[1] >= gold_entity[1]) \
                or (span[0] >= gold_entity[0] and span[1] <= gold_entity[1]):
            if span in copy_potential_extra_entity:
              error_summary["boundary-error"].append({
                "gold_entity_str": " ".join(example["tokens"][gold_entity[0]: gold_entity[1]+1]),
                "gold_entity_span": gold_entity,
                "pred_entity_str": " ".join(example["tokens"][span[0]: span[1]+1]),
                "pred_entity_span": span})
              boundary_error += 1
              copy_potential_extra_entity.remove(span)
              if gold_entity in potential_miss_entity:
                potential_miss_entity.remove(gold_entity)
      for span in copy_potential_extra_entity:
        error_summary["extra-entity-error"].append({
            "pred_entity_str": " ".join(example["tokens"][span[0]: span[1] + 1]),
            "pred_entity_span": span})
        extra_entity_error += 1
      for span in potential_miss_entity:
        error_summary["miss-entity-error"].append({
              "gold_entity_str": " ".join(example["tokens"][span[0]: span[1]+1]),
              "gold_entity_span": span})
        miss_entity_error += 1
    print(json.dumps(error_summary, indent=2))
  total_error_count = type_error + boundary_error + extra_entity_error + miss_entity_error
  print("type-error", type_error, type_error / total_error_count)
  print("boundary-error", boundary_error, boundary_error / total_error_count)
  print("extra-entity-error", extra_entity_error, extra_entity_error / total_error_count)
  print("miss-entity-error", miss_entity_error, miss_entity_error / total_error_count)


if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--task_name", choices=["ner"])
  parser.add_argument("--dump_file_path", type=str)

  args = parser.parse_args()

  if args.task_name == "ner":
    ner_analysis(args.dump_file_path)