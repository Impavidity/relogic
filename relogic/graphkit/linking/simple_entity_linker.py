from typing import Dict, List

from relogic.structures.sentence import Sentence
from relogic.structures.span import Span
from relogic.structures.structure import Structure
from relogic.structures.document import Document
from relogic.structures.linkage_candidate import LinkageCandidate

from relogic.graphkit.utils.similarity_function import jaccard_similarity

"""
A quick hack
"""
import os
os.environ['JAVA_HOME'] = "/usr/lib/jvm/java-8-openjdk-amd64/"
import jnius_config
jnius_config.set_classpath("/data/lctan/Nesoi/Anserini/target/anserini-0.6.0-SNAPSHOT-fatjar.jar")

from jnius import autoclass
JString = autoclass('java.lang.String')
JSearcher = autoclass('io.anserini.search.KGSearcher')
"""
End of quick hack
"""


class SimpleEntityLinker(object):
  """Two step linking.
  
  Args:
    paths (Dict): The key is the name for the retriever, and value is the directory path to
      the Lucene index.
  """
  def __init__(self, paths: Dict):
    self.paths = paths
    self.retrievers = {}
    for name, path in self.paths.items():
      self.retrievers[name] = JSearcher(JString(path))
      
      # self.retrievers[name].setLanguage()

  def entity_retrieval(self, mention: Span, name: str, candidate_size: int):
    hits = self.retrievers[name].search(
      JString(mention.text.encode("utf-8")), JString("contents"), True, None, candidate_size)
    # Retrieve entity linking results from Anserini.
    # The data structure follows the definition in Anserini.
    # LinkageCandidate.from_hit need to follow the changes in Anserini.
    for i in range(len(hits)):
      mention.add_linkage_candidate(LinkageCandidate.from_hit(hits[i]))

  def rank(self, mention: Span):
    """Operate based on span, and change the results in position.
    1. Sorted by retrieval score
    2. Given score range [max_score - epsilon, max_score], aggregate the URIs
      from prior or/and alias_of
    3. Convert the URI to string (process the URI or use the label of URI), and 
      rank with the string # A quick hack for DBpedia or Wikidata.
    """
    if mention.linkage_candidates:
      mention.linkage_candidates = sorted(mention.linkage_candidates,
        key=lambda x: self.similarity(x.text, mention.text))
    else:
      return
    
    max_score = mention.linkage_candidates[0].score


  
  def similarity(self, x, y):
    return jaccard_similarity(x.lower().split(), y.lower().split())

  def global_inference(self, inputs: Structure):
    """The global inference is to deal with mention boundary overlap.
    Due to imperfect entity detection, there will be some uncertain entities, causing
      the overlap of some entities. For example, in sentence 'Who painted The Storm on 
      the Sea of Galilee', 'The Storm on the Sea of Galilee' and 'The Storm' are
      extracted as mentions at the same time due to different entity mention sources are
      aggregated. With global inference, duplicate/incorrect entity such as 'The Storm' 
      will be removed because it is full substring of 'The Storm on the Sea of Galilee'.
      This process can not be done before linking, because we do not know which mention is 
      better solely based on the surface form of the mention. With preliminary entity
      linking results, we can resolve these better.
    """
    pass




  def link(self, inputs: List[Structure]):
    """Linking method.
    If the inputs is Sentence, then two steps linking is operated.
    If the inputs is a Span, then candidate retrieval is operated 
      without global inference.
    """
    for item in inputs:
      if isinstance(item, Sentence) or isinstance(item, Document):
        pass
      elif isinstance(item, Span):
        pass
      else:
        raise ValueError("Item type {} is not supported".format(type(item)))
