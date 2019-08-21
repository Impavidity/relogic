import os
import sys
from relogic.utils.file_utils import cached_path, RELOGIC_CACHE



PACKAGE_PATH = {
  "Anserini": "https://git.uwaterloo.ca/p8shi/data-server/raw/master/anserini-0.6.0-SNAPSHOT-fatjar.jar"
}

anserini_cache_path = cached_path(PACKAGE_PATH['Anserini'], cache_dir=RELOGIC_CACHE)


if sys.platform == 'win32':
  separator = ';'
else:
  separator = ':'

jar = os.path.join(separator + anserini_cache_path)

if 'CLASSPATH' not in os.environ:
  os.environ['CLASSPATH'] = jar
else:
  os.environ['CLASSPATH'] += jar