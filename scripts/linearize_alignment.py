"""Convert the fast_align
New York , 2-27 May 2005 ||| Nueva York , 2 a 27 de mayo de 2005
0-0 1-1 1-2 2-2 3-3 3-4 3-5 3-6 4-7 4-8 5-9

the expected output is
([0, 1, 3, 4, 5], [0, 1, 3, 7, 9])
The algorithm here is to use connect components.
"""
import argparse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import numpy as np
import json

def main(alignment_file_path, parallel_file_path, selected_indices_file_path):
  fout = open(selected_indices_file_path, 'w')
  counter = 0
  with open(alignment_file_path) as fin_align, open(parallel_file_path) as fin_parallel:
    for alignment, sentence_pair in zip(fin_align, fin_parallel):
      text_a, text_b = sentence_pair.strip().split("|||")
      a_tokens, b_tokens = text_a.strip().split(), text_b.strip().split()
      length = len(a_tokens) + len(b_tokens)
      shape = (length, length)
      a_base_index, b_base_index = 0, len(a_tokens)
      text_a_index_array, text_b_index_array, indicator_array = [], [], []
      for pair in alignment.strip().split():
        index_a, index_b = pair.split('-')
        index_a = int(index_a) + a_base_index
        index_b = int(index_b) + b_base_index
        text_a_index_array.append(index_a)
        text_b_index_array.append(index_b)
        indicator_array.append(1)
      matrix = csr_matrix((np.array(indicator_array), (np.array(text_a_index_array), np.array(text_b_index_array))), shape=shape)
      n_components, labels = connected_components(csgraph=matrix, directed=False, return_labels=True)
      a_components, b_components = [], []
      for i in range(n_components):
        a_components.append([])
        b_components.append([])
      for i in range(len(a_tokens)):
        a_components[labels[a_base_index+i]].append(i)
      for i in range(len(b_tokens)):
        b_components[labels[b_base_index+i]].append(i)
      a_selected_index, b_selected_index = [], []
      for a_component, b_component in zip(a_components, b_components):
        if len(a_component) > 0 and len(b_component) > 0:
          a_selected_index.append(min(a_component))
          b_selected_index.append(min(b_component))
      fout.write(json.dumps([a_selected_index, b_selected_index]) + "\n")
      counter += 1
      if counter % 100000 == 0:
        print(counter)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--alignment_file_path", type=str)
  parser.add_argument("--parallel_file_path", type=str)
  parser.add_argument("--selected_indices_file_path", type=str)
  args = parser.parse_args()
  main(args.alignment_file_path, args.parallel_file_path, args.selected_indices_file_path)
