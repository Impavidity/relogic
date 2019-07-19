from relogic.components.component import Component
from relogic.structures.structure import Structure

class EntityLinkingComponent(Component):
  """For given entity mention, this component is to 
  link the mention to a given knowledge graph.
  """
  def __init__(self):
    pass
  
  def execute(self, inputs: Structure):
    """All entity mentions in the inputs are given.
    Retrieve a candidate list regarding each mention.
    Then the global inference is operated to decide the top
      ranked item for each mention.
    """
    