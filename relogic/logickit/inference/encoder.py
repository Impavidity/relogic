from relogic.logickit.inference.modeling import BertPreTrainedModel, BertModel
from relogic.logickit.inference.modeling_xlm import XLMPreTrainedModel, XLMModel
import torch
import torch.nn as nn

def get_encoder(encoder_type):
  if encoder_type == "bert":
    return Encoder
  if encoder_type == "xlm":
    return XLMEncoder
  if encoder_type == "xlmr":
    return XLMRobertaEncoder

class XLMRobertaEncoder(nn.Module):
  def __init__(self, pretrained_model_name_or_path):
    super().__init__()
    self.xlmr = torch.hub.load('pytorch/fairseq', pretrained_model_name_or_path)

  def forward(self, input_ids, **kwargs):
    sequence_output = self.xlmr.extract_features(input_ids)
    return sequence_output

  @classmethod
  def from_pretrained(cls, pretrained_model_name_or_path):
    return cls(pretrained_model_name_or_path)


class XLMEncoder(nn.Module):
  def __init__(self, pretrained_model_name_or_path):
    super(XLMEncoder, self).__init__()
    self.xlm = XLMModel.from_pretrained(
      pretrained_model_name_or_path)

  def forward(self,
              input_ids,
              attention_mask=None,
              langs=None,
              token_type_ids=None,
              position_ids=None,
              lengths=None,
              cache=None,
              head_mask=None,
              output_all_encoder_layers=False,
              selected_non_final_layers=None,
              route_path=None,
              no_dropout=True,
              **kwargs):
    """

    :param input_ids:
    :param attention_mask:
    :param langs: (batch_size, sentence_length)
    :param token_type_ids:
    :param position_ids:
    :param lengths:
    :param cache:
    :param head_mask:
    :param output_all_encoder_layers:
    :param selected_non_final_layers:
    :param route_path:
    :param no_dropout:
    :return:
    """
    sequence_output = self.xlm(input_ids=input_ids,
             attention_mask=attention_mask,
             langs=langs,
             token_type_ids=token_type_ids,
             position_ids=position_ids,
             lengths=lengths,
             cache=cache,
             head_mask=head_mask)
    return sequence_output

  @classmethod
  def from_pretrained(cls, pretrained_model_name_or_path):
    return cls(pretrained_model_name_or_path)




class Encoder(BertPreTrainedModel):
  def __init__(self, config, output_attentions=False):
    super(Encoder, self).__init__(config)
    self._config = config
    self.output_attentions = output_attentions
    self.bert = BertModel(config, output_attentions=output_attentions)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.apply(self.init_bert_weights)

  def forward(self,
              input_ids,
              token_type_ids=None,
              attention_mask=None,
              head_mask=None,
              output_all_encoded_layers=False,
              token_level_attention_mask=None,
              selected_non_final_layers=None,
              route_path=None,
              no_dropout=False,
              **kwargs):
    sequence_output = self.bert(
      input_ids=input_ids,
      token_type_ids=token_type_ids,
      attention_mask=attention_mask,
      head_mask=head_mask,
      token_level_attention_mask=token_level_attention_mask,
      output_all_encoded_layers=output_all_encoded_layers,
      selected_non_final_layers=selected_non_final_layers)
    if self.output_attentions:
      attention_map, sequence_output, _ = sequence_output
    else:
      sequence_output, _ = sequence_output
    if not no_dropout:
      if output_all_encoded_layers or selected_non_final_layers is not None:
        sequence_output = [self.dropout(seq) for seq in sequence_output]
      else:
        sequence_output = self.dropout(sequence_output)
    if self.output_attentions:
      return sequence_output, attention_map
    else:
      return sequence_output


if __name__ == "__main__":
  import torch
  model = BertModel.from_pretrained("bert-large-uncased-whole-word-masking")
  # layer1 = model.encoder.layer[0]
  # print(layer1[0].attention.self.value.weight.data - layer1[1].attention.self.value.weight.data)
  feature, _ = model(torch.zeros(1, 5, dtype=torch.long),
                 output_all_encoded_layers=False)


  print(feature)

  encoder = XLMEncoder("xlm-mlm-tlm-xnli15-1024")
  feature = encoder(input_ids=torch.zeros(1, 5, dtype=torch.long))
  print(feature)
