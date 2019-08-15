import torch
import torch.nn as nn

class SpanGCNModule(nn.Module):
  """
  SpanGCN firstly extract span from text, and then label each span based
    on the learned representation of GCN
  """
  def __init__(self, config, task_name, boundary_n_classes=None, label_n_classes=None):
    super(SpanGCNModule, self).__init__()
    self.config = config
    self.task_name = task_name
    self.boundary_n_classes = boundary_n_classes
    self.label_n_classes = label_n_classes
    if boundary_n_classes:
      self.to_boundary_logits = nn.Linear(config.hidden_size, self.boundary_n_classes)
    if label_n_classes:
      self.to_label_logits = nn.Linear(config.hidden_size * 2, self.label_n_classes)
    if config.use_gcn:
      pass
    else:
      pass
    self.padding = nn.Parameter(torch.zeros(config.hidden_size), requires_grad=False)
    self.ones = nn.Parameter(torch.ones(1, 1), requires_grad=False)

  def forward(self,
              input, predicate_span=None,
              bio_hidden=None, span_candidates=None, extra_args=None):
    """
    Before this module, there is another module info aggregation
    :param input: Sentence Only, in batch
    :param predicate: Predicate Only, in batch
    :param bio_hidden: hidden vector for span prediction, can be None
    :param span_candidates: tuple, span_start, and span_end
    :param extra_args: strategy
    :return: labeling_logits

    Here we need to support three mode of inference
    1. Span is given
       In this mode, sequence labeling and independent span generation modes are supported.
       span_logits = None, span_candidates = batch of span
       Another problem here is how to aggregate span. We need to specify span level and aggregate method
       - For token level span
         - Pooling
         - Average
         - Attentive
         - Head Tail Attentive
       - For phrase level span
         - Non-Hierarchy
           - Pooling
           - Average
           - Attentive
           - Head Tail Attentive
         - Hierarchy
           - Use Token level aggregate
           - Use Non-Hierarchy to aggregate again
    2. Span is not given
       In this mode, dependent span generation mode is supported.
       span_logits = batch, span_candidates = None
       span candidates generate from span_logits. There are two logits need to be return
       This mode only have Phrase level span Aggregation

    After we have span representation, how to interact with predicate
    1. If the span itself is predicate aware, do we need to add predicate information again ?
       So the experiments is on surface form aware or not. You can refer the IBM relation paper
    2. If the span itself is not predicate aware, it will be trained faster.
       How to design a module to interact the argument and the predicate
       - Independent Classifier
       - Bilinear
       - GCN
    """
    if bio_hidden:
      bio_logits = self.to_span_logits(bio_hidden)
      assert "label_mapping" in extra_args, "label_mapping does not in extra_args"
      span_candidates = get_candidate_span(bio_logits, extra_args["label_mapping"])
    start_index, end_index = span_candidates
    # start_index, end_index = (batch, max_span_num)
    predicate_start_index, predicate_end_index = predicate_span
    # predicate_start_index, predicate_end_index = (batch)
    max_span_num = len(start_index[0])
    # input (batch, sentence, dim) -> (batch, max_span_num, sentence, dim)
    expanded_input = input.unsqueeze(1).repeat(1, max_span_num, 1, 1)
    start_index_ = start_index.view(-1)
    end_index_ = end_index.view(-1)

    span_hidden = select_span(expanded_input.view(-1, expanded_input.size(-2), expanded_input.size(-1)), start_index_, end_index_, self.padding)
    predicate_hidden = select_span(input, predicate_start_index, predicate_end_index, self.padding)

    span_repr = self.aggregate(span_hidden, end_index_-start_index_)
    predicate_repr = self.aggregate(predicate_hidden, predicate_end_index-predicate_start_index)
    # (batch, dim)
    concat = torch.cat([span_repr, predicate_repr.unsqueeze(1).repeat(1, max_span_num, 1).view(-1, predicate_repr.size(-1))], dim=-1)
    label_logits = self.to_label_logits(concat)

    return label_logits.view(input.size(0), max_span_num, self.label_n_classes)

  def aggregate(self, hidden, lengths):
    """
    Use average for now
    :param hidden: (batch, span_length, dim)
    :param lengths: (batch)
    :return:
    """
    return torch.sum(hidden, 1) / torch.max(
      self.ones.repeat(lengths.size(0), 1).float(), lengths.unsqueeze(1).float())

def select_span(input, start_index, end_index, padding):
  """
  Use for loop to select
  :param input:
  :param start_index:
  :param end_index:
  :param padding:
  :return:
  """
  padded_tensor = []

  max_span_size = torch.max(end_index - start_index)
  for idx, (start, end) in enumerate(zip(start_index, end_index)):
    padded_tensor.append(
      torch.cat(
        [torch.narrow(input[idx], 0, start, (end-start))] +
        ([padding.unsqueeze(0).repeat(max_span_size-(end-start), 1)] if max_span_size != (end-start)
        else []), dim=0))
    # list of (max_span_size, dim)
  return torch.stack(padded_tensor)

def get_candidate_span(bio_logits, label_mapping):
  """
  Use python for now. Will consider a C++ binding later.
  :param bio_logits: (batch_size, sentence_length, 3)
  :param label_mappings:
  :return:
  """
  preds_tags = bio_logits.argmax(-1).data.cpu().numpy()
  inv_label_mapping = {v: k for k, v in label_mapping.items()}
  batch_span_labels = []
  max_span_num = 0
  for sentence in bio_logits:
    # convert to understandable labels
    sentence_tags = [inv_label_mapping[i] for i in sentence]
    span_labels = []
    last = 'O'
    start = -1
    for i, tag in enumerate(sentence_tags):
      pos, _ = (None, 'O') if tag == 'O' else tag.split('-', 1)
      if (pos == 'S' or pos == 'B' or tag == 'O') and last != 'O':
        span_labels.append((start, i - 1, last.split('-')[-1]))
      if pos == 'B' or pos == 'S' or last == 'O':
        start = i
      last = tag
    if sentence_tags[-1] != 'O':
      span_labels.append((start, len(sentence_tags) - 1,
                          sentence_tags[-1].split('-', 1)[-1]))
    max_span_num = max(len(span_labels), max_span_num)
    batch_span_labels.append(span_labels)
  batch_start_index = []
  batch_end_index = []
  for span_labels in batch_span_labels:
    start_index = []
    end_index = []
    for span_label in span_labels:
      start_index.append(span_label[0])
      end_index.append(span_label[1])
    start_index += (max_span_num - len(start_index)) * [0]
    end_index += (max_span_num - len(end_index)) * [0]
    # Just a placeholder, for loss computation, it will be ignored.
    batch_start_index.append(start_index)
    batch_end_index.append(end_index)
  start_ids = torch.tensor(batch_start_index, dtype=torch.long).to(bio_logits.device)
  end_ids = torch.tensor(batch_end_index, dtype=torch.long).to(bio_logits.device)
  return (start_ids, end_ids)


