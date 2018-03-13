from __future__ import print_function

import os
import os.path as osp
import numpy as np
from PIL import Image as Im
import sys

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from gen_path import gen_path_list

feature_path = "represent.npy"
tsne_path = 'tsne.npy'

vision_img = 'tsne-test.jpg'

X = np.load(feature_path)
n_train_samples = len(X)
nsamples = n_train_samples
# n_train_samples = 5000
# nsamples = 5000


# PCA reduce dimension to accelerate the speed of TSNE
X_pca = PCA(n_components=32).fit_transform(X)
X_train = X_pca[:n_train_samples]

# TSNE dimension reduction
X_train_embedded = TSNE(n_components=2, perplexity=40, verbose=2).fit_transform(X_train)
print(type(X_train_embedded))

np.save(tsne_path, X_train_embedded)

# plot figure
tsne = np.load(tsne_path)
#  you can modify here to random sample
tsne = tsne[:nsamples]
tsne -= tsne.min()
expand_factor = 256
tsne *= expand_factor
size = np.ceil(tsne.max()) + 64
size = int(size)
print("printing poster size : (%d, %d)" % (size, size))

# generate images list
data_folder = "/Users/wangxun/DataSet/CUB_200_2011/test"
path_list = gen_path_list(data_folder)

#  you can modify here to random sample like this
# path_list = path_list[index]

print('there are %05d images in %s' % (len(path_list), data_folder))
images = []
for i, filename in enumerate(path_list):
    img_path = osp.join(data_folder, filename)
    # print(osp.exists(img_path))
    x = Im.open(img_path)
    x = np.array(x)

    # only RGB image is allowed
    if not len(x.shape) == 3:
        x = np.ones((50, 50, 3), dtype=int) * 255
        x = x.astype("uint8")
    images.append(x)
    if i >= nsamples-1:
        break

# poster
bigpic = np.ones((size, size, 3), dtype=int) * 255

for k, (im, coord) in enumerate(zip(images, tsne)):
    print("%.2f %% completed\r" % ((k + 1) * 100. / nsamples))
    sys.stdout.flush()
    x, y = int(coord[0]), int(coord[1])
    im = Im.fromarray(im).resize((50, 50), Im.ANTIALIAS)
    img = np.array(im).astype("uint8")
    bigpic[x:x + 50, y:y + 50, :] = np.array(im).astype("uint8")
    bigpic = bigpic.astype('uint8')

Im.fromarray(bigpic).convert('RGB').save(vision_img)

