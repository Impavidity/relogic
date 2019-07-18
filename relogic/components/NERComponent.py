from relogic.components.component import Component
from relogic.structures.structure import Structure

from relogic.logickit.training.trainer import Trainer

class NERComponent(Component):
  """

  """
  def __init__(self, config, trainer=None):
    super(NERComponent, self).__init__()
    self._trainer = trainer if trainer else Trainer(config=config)

  def execute(self, inputs: Structure):
    self._trainer.predict(inputs, task_name="ner")



