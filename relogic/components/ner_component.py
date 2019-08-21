from relogic.components.component import Component
from relogic.structures.structure import Structure
from typing import List


class NERComponent(Component):
  """

  """
  def __init__(self, config, predictor=None):
    super(NERComponent, self).__init__(config, predictor)

  def execute(self, inputs: List[Structure]):
    self._predictor(inputs)




