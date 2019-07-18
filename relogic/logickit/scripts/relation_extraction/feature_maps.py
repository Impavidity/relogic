import json

import numpy as np
from tqdm import tqdm
import spacy
from spacy.tokens import Doc

import networkx as nx

nlp = spacy.load("en_core_web_lg")


def read_data(path):
  with open(path, 'r') as f:
    data = json.load(f)
  return data


no_find_rdp = []

def build_lines(data):
  lines = []
  for s in data:
    s_start = s['subj_start']
    s_end = s['subj_end'] + 1
    subj = ' '.join(s['token'][s_start:s_end])
    o_start = s['obj_start']
    o_end = s['obj_end'] + 1
    obj = ' '.join(s['token'][o_start:o_end])

    line = {'tokens': s['token'],
            'length': len(s['token']),
            'text': ' '.join(s['token']),
            'subj': s['token'][s_start:s_end],
            'obj': s['token'][o_start:o_end],
            'relation': s['relation']}

    lines.append(line)
  print("Finish convert data to lines")
  return lines


def find_rdp(line, graph, doc, offset=2):
  """
  构建一个dep_pos = {0: [xx, xxx,],
                    1: [xx, xxx,]}
  用于查看不同距离的pairs。
  把dep_pos相关的行注释掉也行，可以省计算
  rdp: 直接保存所有距离在offset以内的token，充当sdp的作用

  """
  rdp = set()
  #     dep_pos = defaultdict(list)

  targets = line['subj'] + line['obj']  # ['Chunghwa', 'Telecom'] + ['Taiwan']

  for target in targets:
    for token in doc:
      try:
        dep_pos_length = nx.shortest_path_length(graph, source=token.text, target=target)
        dep_pos_path = nx.shortest_path(graph, source=token.text, target=target)
        if dep_pos_length <= offset:
          #                     dep_pos[dep_pos_length].append(dep_pos_path)
          rdp.add(token.text)
      except nx.NetworkXNoPath:
        pass
      except nx.NodeNotFound:  #
        no_find_rdp.append(line)

  return list(rdp)


def get_sdps(lines):
  print('Begin getting sdps from lines: ')
  # count how many sample have no sdp
  sdps = []  # change to set
  no_sdps = 0
  no_sdp_samples = []
  no_rdp_samples = []
  for line in tqdm(lines):
    # spacy parser
    doc = Doc(nlp.vocab, words=line['tokens'])

    nlp.tagger(doc)
    nlp.parser(doc)

    # create edges
    edges = []

    for t in doc:
      edges.append((t.text, t.head.text))

    subj_list = line['subj']  # ['Chunghwa', 'Telecom']
    obj_list = line['obj']  # ['Taiwan']

    # build graph
    graph = nx.Graph(edges)

    # get sdp
    sdp = []
    no_sdp = 0
    no_sdp_sample = {'no_sdp_pairs': [], 'text': None}
    for obj in obj_list:
      for subj in subj_list:
        try:
          sdp.append(nx.shortest_path(graph, source=subj, target=obj))
        except nx.NetworkXNoPath:  # if subj -> obj not exist
          no_sdp += 1
          no_sdp_sample['no_sdp_pairs'].append((obj, subj))
          no_sdp_sample['text'] = line['text']

    if no_sdp_sample['text'] is not None:  # no one sdp, so we use relative dependency position instead
      no_sdp_samples.append(no_sdp_sample)
      sdp.append(find_rdp(line, graph, doc, offset=2))
    no_sdps += no_sdp
    sdps.append(sdp)
  print("Finish convert lines to sdps")
  return sdps, no_sdps, no_sdp_samples, no_rdp_samples


def pretty_sdps(old_sdps):
  new_sdps = []
  for old_sdp in old_sdps:  # each sample
    new_sdp = set()
    for pair_path in old_sdp:  # each pair path in one old sdp
      new_sdp = new_sdp.union(set(pair_path))
    new_sdps.append(list(new_sdp))
  return new_sdps


def sdp_map(sdps, lines):
  """
  Get sdp attention map
  input:
      sdps: 记录所有sentence的sdp
      lines：记录所有sentence的meta data
  output:
      sdp_maps: a list contains all sdp maps
  """
  print('Begin getting sdp_map from sdps')
  sdp_maps = []
  for i, line in enumerate(lines):
    sdp_vector = [0] * line['length']
    sdp = sdps[i]
    for j, token in enumerate(line['tokens']):
      if token in sdp:
        sdp_vector[j] = 1

    sdp_vector = np.array(sdp_vector)
    sdp_map = np.dot(sdp_vector.reshape(sdp_vector.size, -1), sdp_vector.reshape(-1, sdp_vector.size))
    sdp_maps.append(sdp_map)
  print("Finish converting sdp to sdp_maps")
  return sdp_maps


def save_map(maps, name='xxx_maps.npy'):
  """
  name: which-map_which-dataset_maps
        sdp_train_maps.npy
  """
  np.save(name, maps)
  print("save data to {}".format(name))


def load_map(name):
  maps = np.load(name, allow_pickle=True)
  print("load data from {}".format(name))
  return maps

# def main():
data = read_data('../../dataset/tacred/dev.json')
lines = build_lines(data)
sdps, no_sdps, no_sdp_samples, no_rdp_samples = get_sdps(lines)
sdps = pretty_sdps(sdps)
sdp_maps = sdp_map(sdps, lines)
save_map(sdp_maps, name="sdp_dev_maps.npy")