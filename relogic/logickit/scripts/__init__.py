from scripts import squad11_eval
from scripts import squad20_eval

def squad_eval(dataset):
  if dataset == "squad11":
    return squad11_eval.evaluate
  if dataset == "squad20":
    return squad20_eval.evaluate