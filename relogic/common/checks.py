class ConfigurationError(Exception):
  """
  The exception raised by any AllenNLP object when it's misconfigured
  (e.g. missing properties, invalid properties, unknown properties).
  """

  def __init__(self, message):
    super().__init__()
    self.message = message

  def __str__(self):
    return repr(self.message)