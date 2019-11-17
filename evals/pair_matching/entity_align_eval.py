import random
import argparse
import json
import numpy as np
import scipy.spatial


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--embd", help="Paht to BERT embedding file. (JSON)")
    p.add_argument("-g", "--gold", help="Paht to gold pairs (ground truth). (TXT)")
    args = p.parse_args()
    return args



def get_hits(vec, test_pair, top_k=(1, 10, 50, 100, 200)):
    Lvec = np.array([vec[e1] for e1, e2 in test_pair])
    Rvec = np.array([vec[e2] for e1, e2 in test_pair])
    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
    top_lr = [0] * len(top_k)
    for i in range(Lvec.shape[0]):
        rank = sim[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    top_rl = [0] * len(top_k)
    for i in range(Rvec.shape[0]):
        rank = sim[:, i].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1
    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print('For each right:')
    for i in range(len(top_rl)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))


def load_gold(path):
    gold = []
    with open(path) as f:
        for line in f:
            spt = line.strip().split("\t")
            gold.append((int(spt[0]), int(spt[1])))
    gold = sorted(gold, key=lambda x:x[0])
    return np.array(gold)


def load_embd(path):
    embd_dict = {}
    with open(path) as f:
        for line in f:
            example = json.loads(line.strip())
            vec = np.array([float(e) for e in example['feature'].split()])
            embd_dict[int(example['guid'])] = vec
    return embd_dict


def main():
    args = parse()
    gold = load_gold(args.gold)
    embd_dict = load_embd(args.embd)
    get_hits(embd_dict, gold)


if __name__ == "__main__":
    main()
