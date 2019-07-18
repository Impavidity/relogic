"""
1. Read raw data
2. Preprocess:
    convert to id
    save length
    padding sequence to max length
    bucket them
3. Save to hd5f
4. Load file with max_memory_limit
5. Shuffle
6. Slice to remove extra paddings
"""

class LazyDataset(object):
  def __init__(self):
    pass