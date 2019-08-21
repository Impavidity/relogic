from typing import Dict, List, Union

from relogic.structures.sentence import Sentence
from relogic.structures.span import Span
from relogic.structures.structure import Structure
from relogic.structures.document import Document
from relogic.structures.linkage_candidate import LinkageCandidate

from relogic.graphkit.utils.similarity_function import jaccard_similarity
from relogic.utils.file_utils import cached_path, RELOGIC_CACHE
import os
from zipfile import ZipFile
import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

"""
A quick hack
"""
from jnius import autoclass
JString = autoclass('java.lang.String')
JSearcher = autoclass('io.anserini.search.KGSearcher')

"""
End of quick hack
"""

INDEX_PATHS = {
  "en_wikipedia": "https://git.uwaterloo.ca/p8shi/data-server/raw/master/entity_linking_index_en.zip",
  "zh_baike": "https://git.uwaterloo.ca/p8shi/data-server/raw/master/entity_linking_index_zh.zip"
}

INDEX_LANGUAGE = {
  "en_wikipedia": "en",
  "zh_baike": "zh"
}

class SimpleEntityLinker(object):
  """Two step linking.
  
  Args:
    paths (Dict): The key is the name for the retriever, and value is the directory path to
      the Lucene index.
  """
  def __init__(self, index_names: List, index_paths: Dict = None, index_language: Dict = None):

    self.index_names = index_names
    self.retrievers = {}
    if index_paths is not None and index_language is not None:
      INDEX_PATHS.update(index_paths)
      INDEX_LANGUAGE.update(index_language)
    for index_name in index_names:
      index_zip_or_dir_path = cached_path(INDEX_PATHS[index_name], cache_dir=RELOGIC_CACHE)
      if os.path.isdir(index_zip_or_dir_path):
        index_path = index_zip_or_dir_path
      else:
        index_path = index_zip_or_dir_path + "." + index_name
      if not os.path.exists(index_path):
        with ZipFile(index_zip_or_dir_path, 'r') as zipObj:
          zipObj.extractall(index_path)
          logger.info("Extract Index {} to {}".format(INDEX_PATHS[index_name], index_path))
      self.retrievers[index_name] = JSearcher(JString(index_path))
      self.retrievers[index_name].setLanguage(INDEX_LANGUAGE[index_name])

  def entity_retrieval(self, mention: Span, name: str, candidate_size: int = 20):
    hits = self.retrievers[name].search(
      JString(mention.text.encode("utf-8")), JString("contents"), True, None, candidate_size)
    # Retrieve entity linking results from Anserini.
    # The data structure follows the definition in Anserini.
    # LinkageCandidate.from_hit need to follow the changes in Anserini.
    for i in range(len(hits)):
      mention.add_linkage_candidate(LinkageCandidate.from_hit(hits[i]))

  def rank(self, mention: Span, epsilon: float = 0.5):
    """Operate based on span, and change the results in position.
    1. Sorted by retrieval score
    2. Given score range [max_score - epsilon, max_score], aggregate the URIs
      from prior or/and alias_of
    3. Convert the URI to string (process the URI or use the label of URI), and 
      rank with the string # A quick hack for DBpedia or Wikidata.

    Args:
      mention (:data:`Span`): Mention with linkage candidates for ranking
      epsilon (float): Control the first layer score range from
        [max_score - epsilon, max_score]
    """
    if mention.linkage_candidates:
      mention.linkage_candidates = sorted(mention.linkage_candidates,
        key=lambda x: self.similarity(x.text, mention.text), reverse=True)
    else:
      return
    
    max_score = mention.linkage_candidates[0].score

    # Aggregate first layer
    # Idempotence operation
    mention.first_layer_prior = {}
    for candidate in mention.linkage_candidates:
      if abs(candidate.score - max_score) < epsilon:
        # Go over the priors
        for uri, count in candidate.prior.items():
          if uri:
            mention.first_layer_prior[uri] = mention.first_layer_prior.get(uri, 0) + count
        # Also consider the alias_of
        for uri in candidate.alias_of:
          if uri:
            mention.first_layer_prior[uri] = mention.first_layer_prior.get(uri, 0) + 1
          # TODO: Here 1 is a hyper parameter

    # Sorted by the prior
    sorted_uris = sorted(mention.first_layer_prior.items(), key= lambda x: x[1], reverse=True)
    # TODO: Consider the two ranking together with weighted sum

    # Hard selection for the exact match
    # TODO: This is hard coded for DBpedia or Wikidata
    mention_text_to_uri = mention.text.lower().replace(" ", "_")
    fixed_sequence = []
    for idx, item in enumerate(sorted_uris):
      # item = (uri, count)
      if mention_text_to_uri == item[0].lower():
        fixed_sequence = [sorted_uris[idx]] + sorted_uris[:idx] + sorted_uris[idx+1:]
        break
    if len(fixed_sequence) > 0:
      mention.ranked_uris = fixed_sequence
    else:
      mention.ranked_uris = sorted_uris

  @staticmethod
  def similarity(x, y):
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


  def link_span(self, span: Span, name: str):
    self.entity_retrieval(span, name=name)
    self.rank(span)

  def link(self, inputs: Union[str, Structure, List[Structure]], name:str):
    """Linking method.
    If the inputs is Sentence, then two steps linking is operated.
    If the inputs is a Span, then candidate retrieval is operated 
      without global inference.
    """
    if isinstance(inputs, str):
      span = Span(text=inputs)
      self.link_span(span, name=name)
      return span
    elif isinstance(inputs, Structure):
      if isinstance(inputs, Document):
        self.link(inputs.sentences, name=name)
      elif isinstance(inputs, Sentence):
        for span in inputs.spans:
          self.link_span(span, name=name)
        self.global_inference(inputs)
      elif isinstance(inputs, Span):
        self.link_span(inputs, name=name)
      else:
        raise ValueError("Item type {} is not supported".format(type(inputs)))
    elif isinstance(inputs, List):
      for item in inputs:
        self.link(item, name=name)
    else:
      raise ValueError("Item type {} is not supported".format(type(inputs)))

