from relogic.logickit.base.constants import *

import torch.nn.functional as F

def get_loss(task_name, logits, label_ids, config, extra_args):
  if task_name in AUTO_SPAN:
    """
    For auto span task, there are two parts in logits:
      one is for span detection
      the other one is for label prediction
    Basically span detection is a sequence labeling task.
      We do not only consider the head of token. So it is 
      full version of sequence labeling
    """
    span_logits, label_logits = logits
    token_label_ids, span_label_ids = label_ids
    assert "segment_ids" in extra_args, "segment_ids should be in extra_args"
    assert "input_mask" in extra_args, "input_mask should be in extra_args"
    # We use segment ids to get sentence labeling
    segment_ids = extra_args["segment_ids"]
    input_mask = extra_args["input_mask"]
    active_loss = (segment_ids[:, :span_logits.size(1)].contiguous().view(-1) == 0) & \
                  (input_mask[:, :span_logits.size(1)].contiguous().view(-1) == 1)
    # This operation ensure that it is first part of the input
    active_logits = span_logits.view(-1, span_logits.size(-1))[active_loss]
    active_labels = token_label_ids[:, :span_logits.size(1)].contiguous().view(-1)[active_loss]
    token_loss = F.cross_entropy(active_logits, active_labels)

    assert "span_mask" in extra_args, "span_mask should be in extra_args"
    span_mask = extra_args["span_mask"]
    active_loss = span_mask.view(-1) == 1
    active_logits = label_logits.view(-1, label_logits.size(-1))[active_loss]
    active_labels = span_label_ids.view(-1)[active_loss]
    span_loss = F.cross_entropy(active_logits, active_labels)

    return token_loss + span_loss



