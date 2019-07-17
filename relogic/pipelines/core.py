from relogic.structures.structure import Structure

from typing import List


class Pipeline(object):
  """A pipeline model that run different components.


  """

  def __init__(self, component_names: List[str]):
    self.component_names = component_names
    self.components = {}

  def execute(self, inputs: Structure):
    """Execute the pipeline.
    """
    for component_name in self.component_names:
      component = self.components.get(component_name, None)
      if component is not None:
        component.execute(inputs)
    return inputs

