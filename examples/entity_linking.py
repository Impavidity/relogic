from relogic.graphkit.linking.simple_entity_linker import SimpleEntityLinker
from relogic.structures.span import Span

linker = SimpleEntityLinker({
  "en": "/data/lctan/Nesoi/indexes/lucene-index.en-mention-table.pos+docvectors+rawdocs/"})
span = Span(text="florida")
linker.link([span])

print(span.ranked_uris[0])
