from typing import List

def jaccard_similarity(x: List, y: List):
  intersection = len(list(set(x).intersection(y)))
  union = (len(x) + len(y)) - intersection
  return float(intersection / union)