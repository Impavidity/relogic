from relogic.components.component import Component
from relogic.structures.structure import Structure
from typing import List


class NERComponent(Component):
  """

  """
  def execute(self, inputs: List[Structure]):
    self._predictor(inputs)




