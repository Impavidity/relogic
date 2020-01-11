from relogic.pipelines.core import Pipeline
import relogic.utils.crash_on_ipy

pipeline = Pipeline(
  component_names=["predicate_detection", "srl"],
  component_model_names= {"predicate_detection" : "spacy" ,"srl": "srl-conll12"})

from relogic.structures.sentence import Sentence

sent = Sentence(
  text="The Lexington-Concord Sesquicentennial half dollar is a fifty-cent piece struck by the United States Bureau of the Mint in 1925 as a commemorative coin in honor of the 150th anniversary of the Battles of Lexington and Concord.")

pipeline.execute([sent])
raise NotImplementedError()