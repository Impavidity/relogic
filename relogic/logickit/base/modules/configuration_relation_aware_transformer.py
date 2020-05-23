

class RelationAwareTransformerConfig(object):
  def __init__(self, **kwargs):
    self.output_attentions = kwargs.pop("output_attentions", False)
    self.output_hidden_states = kwargs.pop("output_hidden_states", False)

    self.is_decoder = kwargs.pop("is_decoder", False)
    self.hidden_size = kwargs.pop("hidden_size", 2048)
    self.num_hidden_layers = kwargs.pop("num_hidden_layers", 8)
    self.num_attention_heads = kwargs.pop("num_attention_heads", 8)
    self.hidden_act = kwargs.pop("hidden_act", "relu")

    self.intermediate_size = kwargs.pop("intermediate_size", 1024)
    self.hidden_dropout_prob = kwargs.pop("hidden_dropout_prob", 0.1)
    self.layer_norm_eps = kwargs.pop("layer_norm_eps", 1e-12)

    self.attention_probs_dropout_prob = kwargs.pop("attention_probs_dropout_prob", 0.1)



