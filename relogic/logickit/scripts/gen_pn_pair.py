import argparse
import os


def main(input_folder, output_folder):
  for fold in ["train"]:
    path = os.path.join(input_folder, fold+".txt")
    target_path = os.path.join(output_folder, fold+".txt")
    left_to_right = {}
    right_to_left = {}
    left = {}
    right = {}
    with open(path) as fin:
      for line in fin:
        if line.startswith("Left-Right-Query"):
          continue
        id_pair, comm_split1, comm_split2, label, gold_id_pair = line.strip().split('\t')
        split1, split2, direc = id_pair.split('-')
        if direc == "0":
          if split1 not in left_to_right:
            left_to_right[split1] = [None, []]
          if label == '1':
            left_to_right[split1][0] = split2
          else:
            left_to_right[split1][1].append(split2)
          left[split1] = comm_split1
          right[split2] = comm_split2
        if direc == "1":
          if split2 not in right_to_left:
            right_to_left[split2] = [None, []]
          if label == '1':
            right_to_left[split2][0] = split1
          else:
            right_to_left[split2][1].append(split1)
          left[split1] = comm_split1
          right[split2] = comm_split2
    with open(target_path, 'w') as fout:
      for split1 in left_to_right.keys():
        text_a = left[split1] # query
        gold_can = left_to_right[split1][0] # correct candidate
        text_b = right[gold_can]
        for neg in left_to_right[split1][1]:
          pair_id = "{}-{}-0".format(split1, neg)
          gold_id = "{}-{}-0".format(split1, gold_can)
          text_c = right[neg]
          fout.write("{}\t{}\t{}\t{}\t{}\n".format(
            pair_id, text_a, text_b, text_c, gold_id))
      for split2 in right_to_left.keys():
        text_a = right[split2]
        gold_can = right_to_left[split2][0]
        text_b = left[gold_can]
        for neg in right_to_left[split2][1]:
          pair_id = "{}-{}-0".format(neg, split2)
          gold_id = "{}-{}-0".format(gold_can, split2)
          text_c = left[neg]
          fout.write("{}\t{}\t{}\t{}\t{}\n".format(
            pair_id, text_a, text_b, text_c, gold_id))
  for fold in ["dev", "test"]:
    path = os.path.join(input_folder, fold + ".txt")
    target_path = os.path.join(output_folder, fold + ".txt")
    left = {}
    right = {}
    left_to_right_gold = {}
    right_to_left_gold = {}
    with open(path) as fin:
      for line in fin:
        if line.startswith("Left-Right-Query"):
          continue
        id_pair, comm_split1, comm_split2, label, gold_id_pair = line.strip().split('\t')
        split1, split2, direc = id_pair.split('-')
        if direc == "0":
          left[split1] = comm_split1
          right[split2] = comm_split2
        else:
          left[split1] = comm_split1
          right[split2] = comm_split2
        split1, split2, direc = gold_id_pair.split('-')
        if direc == "0":
          left_to_right_gold[split1] = split2
        else:
          right_to_left_gold[split2] = split1
    with open(target_path, "w") as fout:
      for split1 in left_to_right_gold:
        guid = split1 + "-none-0"
        gold_id = split1 + "-{}-0".format(left_to_right_gold[split1])
        fout.write("{}\t{}\t{}\n".format(
          guid, left[split1], gold_id))
      for split2 in right_to_left_gold:
        guid = "none-" + split2 + "-1"
        gold_id = right_to_left_gold[split2] + "-{}-1".format(split2)
        fout.write("{}\t{}\t{}\n".format(
          guid, right[split2], gold_id))



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_folder")
  parser.add_argument("--output_folder")
  args = parser.parse_args()
  if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)
  main(args.input_folder, args.output_folder)