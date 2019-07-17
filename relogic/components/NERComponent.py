from relogic.components.component import Component
from relogic.structures.structure import Structure

class NERComponent(Component):
  """

  """
  def __init__(self):
    super(NERComponent, self).__init__()
    self._trainer = None

  def execute(self, inputs: Structure):
    batch = []
    for i, b in enumerate(batch):
      self._trainer.predict(b)



