import argparse
from sklearn.decomposition import PCA
import numpy as np

pca = PCA(2)

parser = argparse.ArgumentParser()
parser.add_argument("--vector_file_paths")
args = parser.parse_args()

files = args.vector_file_paths.split(",")
Xs = []
ys = []
for idx, file in enumerate(files):
  fin = open(file, 'rb')
  data = np.load(fin)
  Xs.append(data['X'])
  if ['y'] in data.files:
    ys.append(data['y'])
  else:
    ys.append(np.ones(data['X'].shape[0], int) * idx)
X = np.concatenate(Xs, 0)
y = np.concatenate(ys, 0)

X_proj = pca.fit_transform(X)

import matplotlib.pyplot as plt


plt.scatter(X_proj[:,0], X_proj[:,1],c=y)
plt.savefig("test.png")

print(pca.components_)
print(pca.components_.shape)