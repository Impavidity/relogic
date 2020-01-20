from relogic.components.component import Component
from relogic.structures.structure import Structure
from typing import List


class SRLComponent(Component):
  """

  """

  def execute(self, inputs: List[Structure]):
    counter = 0
    expanded_inputs = []
    mapping = []
    for idx, structure in enumerate(inputs):
      for predicate_index, predicate_text in structure.predicates:
        expanded_inputs.append(
          structure.__class__(
            tokens=structure.tokens, predicate_text=predicate_text, predicate_index=predicate_index))
        mapping.append(idx)
    for results in self._predictor.predict(expanded_inputs):
      _, batch_labels, _ = results
      for labels in batch_labels:
        inputs[mapping[counter]].srl_labels.append(list(labels))
        counter += 1