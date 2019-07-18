"""
CoNLL English Example:

1	But	but	but	CC	CC	_	_	33	33	DEP	DEP	_	_	_	_	_	_	_
2	while	while	while	IN	IN	_	_	33	33	ADV	ADV	_	_	_	_	AM-ADV	_	_
3	the	the	the	DT	DT	_	_	7	7	NMOD	NMOD	_	_	_	_	_	_	_
4	New	new	new	NNP	NNP	_	_	5	5	NAME	NAME	_	_	_	_	_	_	_
5	York	york	york	NNP	NNP	_	_	7	7	NAME	NAME	_	_	_	_	_	_	_
6	Stock	stock	stock	NNP	NNP	_	_	7	7	NAME	NAME	_	_	_	_	_	_	_
7	Exchange	exchange	exchange	NNP	NNP	_	_	8	8	SBJ	SBJ	_	_	A1	_	_	_	_
"""
import argparse

def read_conll(path):
  with open(path) as fin:
    examples = []
    sentence = []
    for line in fin:
      line = line.strip()
      if len(line) == 0:
        examples.append(sentence)
        sentence = []
      else:
        sentence.append(line.split('\t'))
    if len(sentence) > 0:
      examples.append(sentence)
    return examples

def label_process(label):
  if label == '_':
    return 'O'
  else:
    return label.split('.')[1]

def output_sense(examples, output_path):
  with open(output_path, 'w') as fout:
    for example in examples:
      if len(example[0]) <= 14:
        continue
      for items in example:
        fout.write("{} {}\n".format(items[1], label_process(items[13])))
      fout.write("\n")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--conll_path")
  parser.add_argument("--output_path")
  args = parser.parse_args()
  examples = read_conll(args.conll_path)
  output_sense(examples, args.output_path)