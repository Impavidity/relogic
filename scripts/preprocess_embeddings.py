import argparse
import io
import os
import collections
from tqdm import tqdm
import random
import numpy as np

def load_vectors(fname):
  fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
  n, d = map(int, fin.readline().split())
  data = []
  vocab= ["<PAD>", "<UNK>"]
  data.append([random.uniform(-0.1, 0.1) for i in range(d)])
  data.append([random.uniform(-0.1, 0.1) for i in range(d)])
  index = 2
  for line in tqdm(fin):
    tokens = line.rstrip().split(' ')
    data.append(list(map(float, tokens[1:])))
    vocab.append(tokens[0])
    index += 1

  return vocab, data

def main(vec_file_path, vocab_file_path, vector_file_path):
  vocab, data = load_vectors(vec_file_path)
  with open(vocab_file_path, 'w') as fout:
    for token in vocab:
      fout.write(token + "\n")
  np.save(vector_file_path, np.asarray(data))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--vec_file_path", type=str)
  parser.add_argument("--vocab_file_path", type=str)
  parser.add_argument("--vector_file_path", type=str)
  args = parser.parse_args()
  main(args.vec_file_path, args.vocab_file_path, args.vector_file_path)