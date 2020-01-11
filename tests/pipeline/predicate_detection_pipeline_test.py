from relogic.pipelines.core import Pipeline
import relogic.utils.crash_on_ipy

pipeline = Pipeline(
  component_names=["predicate_detection"],
  component_model_names= {"predicate_detection": "spacy"})

from relogic.structures.sentence import Sentence

sent = Sentence(
  text="Barack Obama went to Paris .")

pipeline.execute([sent])
raise NotImplementedError()