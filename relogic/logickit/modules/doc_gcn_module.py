import torch
import torch.nn as nn
import torch.nn.functional as F
from relogic.logickit.utils import utils
from relogic.logickit.modules.span_extractors.average_span_extractor import AverageSpanExtractor
import dgl
dgl.random.seed(1234)
from dgl.nn.pytorch.conv.gatconv import GATConv

class GAT(nn.Module):
  def __init__(self,
               input_dim,
               hidden_dim,
               num_layers,
               heads,
               feat_drop,
               attn_drop,
               negative_slope,
               residual):
    super().__init__()
    self.num_layers = num_layers
    self.activation = F.elu
    self.gat_layers = nn.ModuleList()
    self.gat_layers.append(GATConv(
      in_feats=input_dim,
      out_feats=hidden_dim,
      num_heads=heads[0],
      feat_drop=feat_drop,
      attn_drop=attn_drop,
      negative_slope=negative_slope,
      residual=False,
      activation=self.activation))

    for l in range(1, num_layers):
      self.gat_layers.append(GATConv(
        in_feats=hidden_dim * heads[l-1],
        out_feats=hidden_dim,
        num_heads=heads[l],
        feat_drop=feat_drop,
        attn_drop=attn_drop,
        negative_slope=negative_slope,
        residual=residual,
        activation=self.activation))


  def forward(self, g, features):
    h = features
    for l in range(self.num_layers):
      h = self.gat_layers[l](g, h).flatten(1)
    return h

class DocGCN(nn.Module):
  def __init__(self, config, task_name, n_classes):
    super().__init__()
    self.config = config
    self.average_span_extractor = AverageSpanExtractor(input_dim=config.hidden_size)
    self.gated_gcn = GAT(
      input_dim=config.hidden_size,
      hidden_dim=config.hidden_size,
      num_layers=3,
      heads=[1] * config.gcn_layer_num,
      feat_drop=0,
      attn_drop=0,
      negative_slope=0.2,
      residual=False)

  def extract_token_features(self, features, token_span_indices, features_mask, token_span_indices_mask):
    token_repr = self.average_span_extractor(
      sequence_tensor=features,
      span_indices=token_span_indices,
      sequence_mask=features_mask,
      span_indices_mask=token_span_indices_mask)
    # (batch_size, new_seq_length)
    return token_repr

  def select_nodes(self, features, selected_node_indices):

    selected_nodes = utils.batched_index_select_tensor(features, selected_node_indices)
    return selected_nodes

  def create_list_of_graphs(self, doc_sent_spans, selected_node_indices_mask, edge_data_list):
    graphs = []
    for span, edge_data in zip(doc_sent_spans, edge_data_list):
      doc_selected_node_size = selected_node_indices_mask[span[0]:span[1]].view(-1).sum().int().item()
      g = dgl.DGLGraph()
      g.add_nodes(doc_selected_node_size)
      if len(edge_data) > 0:
        src, dst = tuple(zip(*edge_data))
        g.add_edges(src, dst)
      new_g = dgl.transform.add_self_loop(g)
      graphs.append(new_g)
    return graphs

  def forward(self, *inputs, **kwargs):
    features = kwargs.pop("features")
    extra_args = kwargs.pop("extra_args")
    # [CLS] description [SEP] text [SEP]
    token_spans = kwargs.pop("token_spans")
    masks = kwargs.pop("input_mask")
    selected_indices = kwargs.pop("selected_indices")
    selected_indices_mask = selected_indices >= 0
    selected_indices[selected_indices < 0] = 0
    token_span_indices_mask = token_spans[:, :, 1] > 0
    token_features = self.extract_token_features(
      features=features,
      token_span_indices=token_spans,
      features_mask=masks,
      token_span_indices_mask=token_span_indices_mask)
    selected_features = self.select_nodes(
      features=token_features,
      selected_node_indices=selected_indices)
    # Assume we have the index to be collected, which are numbered.
    # We have the edge between these nodes
    # Collect features for each document. The result is list of tensors.
    edge_data = extra_args.pop("edge_data")
    doc_sent_spans = extra_args.pop("doc_span")

    list_of_graphs = self.create_list_of_graphs(
      doc_sent_spans=doc_sent_spans,
      selected_node_indices_mask=selected_indices_mask,
      edge_data_list=edge_data)

    graph_batch = dgl.batch(list_of_graphs)
    # The return type is dgl.BatchedDGLGraph

    doc_features = selected_features.view(-1, selected_features.size(-1))
    doc_node_data = doc_features[selected_indices_mask.view(-1)]
    graph_node_repr = self.gated_gcn(graph_batch, doc_node_data)

    graph_batch.ndata["x"] = graph_node_repr
    avg = dgl.mean_nodes(graph_batch, "x")

    # Aggregate query feature
    segment_id = kwargs.pop("segment_ids")
    is_head = kwargs.pop("input_head")
    query_mask = ((is_head != 2) * (segment_id == 0) * masks).float().unsqueeze(-1)
    all_query_features = (features * query_mask).sum(1)
    query_features = self.average_span_extractor(all_query_features, doc_sent_spans).squeeze(0)

    distance = F.pairwise_distance(query_features, avg, p=1)

    return distance


