import os
import sys


if sys.platform == 'win32':
  separator = ';'
else:
  separator = ':'

tdbquery_jar = os.path.join(separator + os.path.dirname(os.path.realpath(__file__)), "../resource/jars/tdbquery.jar")