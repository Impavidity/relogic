class PrefixMap(object):
  def __init__(self, mapping):
    self.mapping = mapping

  def __getitem__(self, item):
    selected_key, data = None, None
    for key in self.mapping:
      if item.startswith(key):
        if data is None:
          data = self.mapping[key]
        else:
          raise ValueError("{} is a common prefix of {} and {}".format(item, selected_key, key))
    if data is None:
      raise ValueError("{} is not in the map".format(item))
    return data
