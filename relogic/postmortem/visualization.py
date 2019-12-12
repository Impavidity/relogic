import argparse
from sklearn.decomposition import PCA
import numpy as np

pca = PCA(2)
RS = 123
from sklearn.manifold import TSNE


parser = argparse.ArgumentParser()
parser.add_argument("--vector_file_paths")
parser.add_argument("--output_file_path")
args = parser.parse_args()

files = args.vector_file_paths.split(",")
Xs = []
ys = []
boundary = []
l = 0
for idx, file in enumerate(files):
  fin = open(file, 'rb')
  data = np.load(fin)
  Xs.append(data['X'])
  if ['y'] in data.files:
    ys.append(data['y'])
  else:
    ys.append(np.ones(data['X'].shape[0], int) * idx)
  boundary.append((l, l+len(data['X'])))
  l += len(data['X'])
X = np.concatenate(Xs, 0)
y = np.concatenate(ys, 0)

X_PCA = pca.fit_transform(X)
X_proj = fashion_tsne = TSNE(random_state=RS).fit_transform(X_PCA)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, len(files) + 1)
for i in range(len(files)):
  b = boundary[i]
  axes[i].scatter(X_proj[b[0]:b[1],0], X_proj[b[0]:b[1], 1], c=y[b[0]:b[1]], s=0.5)

axes[-1].scatter(X_proj[:,0], X_proj[:, 1], c=y, s=0.5)
plt.savefig(args.output_file_path)

print(pca.components_)
print(pca.components_.shape)