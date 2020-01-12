from relogic.pipelines.core import Pipeline
import relogic.utils.crash_on_ipy

pipeline = Pipeline(
  component_names=["predicate_detection"],
  component_model_names= {"predicate_detection": "spacy"})

from relogic.structures.sentence import Sentence

sent1 = Sentence(
  text="Barack Obama can't go to Paris.")
sent2 = Sentence(
  text="Pulmonary alveolar proteinosis (PAP) is a rare lung disorder characterized by an abnormal accumulation of surfactant-derived lipoprotein compounds within the alveoli of the lung."
)

pipeline.execute([sent1, sent2])
print(sent1)
print(sent2)