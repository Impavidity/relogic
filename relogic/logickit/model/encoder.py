from relogic.logickit.model.modeling import BertPreTrainedModel, BertModel

import torch.nn as nn

class Encoder(BertPreTrainedModel):
  def __init__(self, config):
    super(Encoder, self).__init__(config)
    self._config = config
    self.bert = BertModel(config, output_attentions=config.output_attention)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.apply(self.init_bert_weights)

  def forward(self,
              input_ids,
              token_type_ids=None,
              attention_mask=None,
              output_all_encoded_layers=False,
              token_level_attention_mask=None,
              selected_non_final_layers=None,
              route_path=None,
              no_dropout=False):
    attention_map, sequence_output, _ = self.bert(
      input_ids=input_ids,
      token_type_ids=token_type_ids,
      attention_mask=attention_mask,
      head_mask=None,
      token_level_attention_mask=token_level_attention_mask,
      output_all_encoded_layers=output_all_encoded_layers,
      selected_non_final_layers=selected_non_final_layers)
    if not no_dropout:
      if output_all_encoded_layers or selected_non_final_layers is not None:
        sequence_output = [self.dropout(seq) for seq in sequence_output]
      else:
        sequence_output = self.dropout(sequence_output)
    return sequence_output, attention_map

if __name__ == "__main__":
  import torch
  model = BertModel.from_pretrained("bert-large-uncased-whole-word-masking")
  # layer1 = model.encoder.layer[0]
  # print(layer1[0].attention.self.value.weight.data - layer1[1].attention.self.value.weight.data)
  feature, _ = model(torch.zeros(1, 5, dtype=torch.long),
                 output_all_encoded_layers=False)
  print(feature)