from relogic.logickit.base.constants import *
from relogic.logickit.tasks.task import Task

import torch.nn.functional as F
import torch

def get_loss(task: Task, logits, label_ids, input_head, config, extra_args, **kwargs):
  if task.name.startswith(IR_TASK):
    return F.cross_entropy(logits, label_ids)
  if task.name in ["joint_srl"]:
    if isinstance(label_ids, tuple):
      label_ids, pred_span_label, arg_span_label, pos_tag_ids = label_ids
    batch_size = label_ids.size(0)
    key = label_ids[:, :, :4]
    batch_id = torch.arange(0, batch_size).unsqueeze(1).unsqueeze(1).repeat(1, key.size(1), 1).to(key.device)
    expanded_key = torch.cat([batch_id, key], dim=-1)
    v = label_ids[:, :, 4]
    # batch_id = torch.arange(0, v.size(0)).unsqueeze(1).repeat(1, v.size(1)).to(key.device)
    # expanded_v = torch.cat([batch_id.unsqueeze(-1), v.unsqueeze(-1)], dim=-1)
    flatten_key = expanded_key.view(-1, 5)
    flatten_v = v.view(-1)

    (srl_scores, top_pred_spans, top_arg_spans, top_pred_span_mask, top_arg_span_mask,
     pred_span_mention_full_scores, arg_span_mention_full_scores, pos_tag_logits) = logits
    # (batch_size, max_pred_num, 2), (batch_size, max_arg_num, 2)
    max_pred_num = top_pred_spans.size(1)
    max_arg_num = top_arg_spans.size(1)
    expanded_top_pred_spans = top_pred_spans.unsqueeze(2).repeat(1, 1, max_arg_num, 1)
    expanded_top_arg_spans = top_arg_spans.unsqueeze(1).repeat(1, max_pred_num, 1, 1)
    indices = torch.cat([expanded_top_pred_spans, expanded_top_arg_spans], dim=-1)
    batch_id = torch.arange(0, batch_size).unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, *indices.size()[1:3], 1).to(
      indices.device)
    expanded_indices = torch.cat([batch_id, indices], dim=-1)
    flatten_expanded_indices = expanded_indices.view(-1, 5)

    # Generate Loss Mask
    flatten_expanded_top_pred_span_mask = top_pred_span_mask.unsqueeze(2).repeat(1, 1, max_arg_num).view(-1)
    # (batch_size, max_pred_to_keep)
    flatten_expanded_top_arg_span_mask = top_arg_span_mask.unsqueeze(2).repeat(1, max_pred_num, 1).view(-1)
    # (batch_size, max_arg_to_keep)
    merged_mask = flatten_expanded_top_pred_span_mask & flatten_expanded_top_arg_span_mask

    # build dictionary
    d = {}
    for key, value in zip(flatten_key.cpu().numpy(), flatten_v.cpu().numpy()):
      d[tuple(key)] = value

    label_list = []
    for index in flatten_expanded_indices.cpu().numpy():
      label_list.append(d.get(tuple(index), 0))

    # arg_boundary = max(torch.max(top_arg_spans).item(), torch.max(key[:,:,2:]).item()) + 1
    # pred_boundary= max(torch.max(top_pred_spans).item(), torch.max(key[:,:,:2]).item()) + 1
    # size = (batch_size, pred_boundary, pred_boundary, arg_boundary, arg_boundary)
    #
    # dense_label = torch.sparse.LongTensor(flatten_key.t(), flatten_v, size).to(key.device)
    #
    # selected_label = dense_label.masked_select(expanded_indices)

    selected_label = torch.LongTensor(label_list).to(label_ids.device)

    label_loss = F.cross_entropy(srl_scores.view(-1, srl_scores.size(-1))[merged_mask == 1], selected_label[merged_mask == 1])

    # Compute the unary scorer loss
    if hasattr(task.config, "srl_candidate_loss") and task.config.srl_candidate_loss:
      flatten_pred_span_mention_full_scores = F.sigmoid(pred_span_mention_full_scores).view(-1)
      flatten_arg_span_mention_full_scores = F.sigmoid(arg_span_mention_full_scores).view(-1)
      flatten_pred_span_label = pred_span_label.view(-1).float()
      flatten_arg_span_label = arg_span_label.view(-1).float()
      srl_pred_candidate_loss = F.binary_cross_entropy(flatten_pred_span_mention_full_scores, flatten_pred_span_label)
      srl_arg_candidate_loss = F.binary_cross_entropy(flatten_arg_span_mention_full_scores, flatten_arg_span_label)
      candidate_loss = srl_pred_candidate_loss + srl_arg_candidate_loss
      return candidate_loss + label_loss

    if hasattr(task.config, "srl_compute_pos_tag_loss") and task.config.srl_compute_pos_tag_loss:
      active_loss = input_head[:, :pos_tag_logits.size(1)].contiguous().view(-1) == 1
      # I use pos_tag_logits.size(1) to get the label length. It is OK to filter extra things
      active_pos_tag_logits = pos_tag_logits.view(-1, pos_tag_logits.size(-1))[active_loss]
      active_labels = pos_tag_ids[:, :pos_tag_logits.size(1)].contiguous().view(-1)[active_loss]
      loss = F.cross_entropy(active_pos_tag_logits, active_labels)
      label_loss += loss

    return label_loss
  elif task.name in [NER_TASK, POS_TASK]:
    active_loss = input_head[:, :logits.size(1)].contiguous().view(-1) == 1
    # I use logits.size(1) to get the label length. It is OK to filter extra things
    active_logits = logits.view(-1, logits.size(-1))[active_loss]
    active_labels = label_ids[:, :logits.size(1)].contiguous().view(-1)[active_loss]
    loss = F.cross_entropy(active_logits, active_labels)
    return loss
  elif task.name in [ENTITY_TYPE_CLASSIFICATION]:
    loss = F.binary_cross_entropy_with_logits(logits, label_ids.float())
    return loss

  elif task.name in [PARALLEL_TEACHER_STUDENT_TASK]:
    active_loss = kwargs.pop("mask")
    if active_loss is not None:
      active_loss = kwargs.pop("mask").view(-1)
      target = kwargs.pop("target")
      active_logits = F.softmax(logits.view(-1, logits.size(-1)), -1)[active_loss]
      active_target = F.softmax(target.view(-1, target.size(-1)), -1)[active_loss]
      loss = F.kl_div(active_logits.log(), active_target)
    else:
      target = kwargs.pop("target")
      if task.config.use_cosine_loss:
        loss = (1 - F.cosine_similarity(logits, target)).sum(-1) / logits.size(0)
      else:
        loss = F.l1_loss(logits, target)
    return loss
  elif task.name in [MIXSENT_TASK]:
    target = kwargs.pop("target")
    # loss = F.mse_loss(logits, target)
    active_logits = F.softmax(logits.view(-1, logits.size(-1)), -1)
    active_target = F.softmax(target.view(-1, target.size(-1)), -1)
    loss = F.kl_div(active_logits.log(), active_target, reduction="batchmean")
    return loss
  else:
    span_boundary, logits = logits
    return F.cross_entropy(logits.view(-1, logits.size(-1)), label_ids.view(-1))

# def get_loss(task_name, logits, label_ids, config, extra_args):
#   if task_name in AUTO_SPAN:
#     """
#     For auto span task, there are two parts in logits:
#       one is for span detection
#       the other one is for label prediction
#     Basically span detection is a sequence labeling task.
#       We do not only consider the head of token. So it is
#       full version of sequence labeling
#     """
#     span_logits, label_logits = logits
#     token_label_ids, span_label_ids = label_ids
#     assert "segment_ids" in extra_args, "segment_ids should be in extra_args"
#     assert "input_mask" in extra_args, "input_mask should be in extra_args"
#     # We use segment ids to get sentence labeling
#     segment_ids = extra_args["segment_ids"]
#     input_mask = extra_args["input_mask"]
#     active_loss = (segment_ids[:, :span_logits.size(1)].contiguous().view(-1) == 0) & \
#                   (input_mask[:, :span_logits.size(1)].contiguous().view(-1) == 1)
#     # This operation ensure that it is first part of the input
#     active_logits = span_logits.view(-1, span_logits.size(-1))[active_loss]
#     active_labels = token_label_ids[:, :span_logits.size(1)].contiguous().view(-1)[active_loss]
#     token_loss = F.cross_entropy(active_logits, active_labels)
#
#     assert "span_mask" in extra_args, "span_mask should be in extra_args"
#     span_mask = extra_args["span_mask"]
#     active_loss = span_mask.view(-1) == 1
#     active_logits = label_logits.view(-1, label_logits.size(-1))[active_loss]
#     active_labels = span_label_ids.view(-1)[active_loss]
#     span_loss = F.cross_entropy(active_logits, active_labels)
#
#     return token_loss + span_loss



