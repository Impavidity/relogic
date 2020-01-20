from relogic.structures.structure import Structure
from relogic.structures.sentence import Sentence
from relogic.components import Component
from relogic.components.ner_component import NERComponent
from relogic.components.entity_linking_component import EntityLinkingComponent
from relogic.components.srl_component import SRLComponent
from relogic.components.predicate_detection_component import PredicateDetectionComponent
from relogic.pipelines.constants import *
from typing import List, Dict

NAME_TO_COMPONENT_CLASS = {
  NER: NERComponent,
  ENTITY_LINKING: EntityLinkingComponent,
  SRL: SRLComponent,
  PREDICATE_DETECTION: PredicateDetectionComponent
}



class Pipeline(object):
  """A pipeline model that run different components.


  """

  def __init__(self,
               component_names: List[str],
               component_model_names: Dict = None,
               component_classes: Dict = None):
    self.component_names = component_names
    if component_classes is not None:
      NAME_TO_COMPONENT_CLASS.update(component_classes)
    self.components = {}
    for component_name in component_names:
      component_class = NAME_TO_COMPONENT_CLASS.get(component_name, None)
      if component_class is not None:
        self.components[component_name] = component_class.from_pretrained(
          pretrained_model_name_or_path=component_model_names[component_name])


  def execute(self, inputs: List[Structure]):
    """Execute the pipeline.
    """
    for component_name in self.component_names:
      component : Component = self.components.get(component_name, None)
      if component is not None:
        component.execute(inputs)
    return inputs

  def __call__(self, inputs):
    if not isinstance(inputs, List) and isinstance(inputs, Structure):
      return self.execute([inputs])
    elif isinstance(inputs, List[Structure]):
      return self.execute(inputs)
    elif isinstance(inputs, str):
      return self.execute([Sentence(text=inputs)])


