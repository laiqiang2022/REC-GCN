import pandas as pd
import numpy as np
# import h5py
from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize

topk = 20

def construct_graph(features, label, method='heat'):
    fname = 'graph/hhar2000_graph_withLabels.txt'
    num = len(label)
    dist = None

    if method == 'heat':
        dist = -0.5 * pair(features) ** 2
        dist = np.exp(dist)
    elif method == 'cos':
        features[features > 0] = 1
        dist = np.dot(features, features.T)
    elif method == 'ncos':
        features[features > 0] = 1
        features = normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)

    inds = []      #是输入0显示多少行，也就是样本个数
    for i in range(dist.shape[0]): #第i行全部元素
        ind = np.argpartition(dist[i, :], -(topk+1))[-(topk+1):]
        inds.append(ind)

        f = open(fname, 'w')
    counter = 0
    A = np.zeros_like(dist)
    for i, v in enumerate(inds):
        mutual_knn = False
        for vv in v:
            if vv == i:
                pass
            else:
                if label[vv] != label[i]:
                    counter += 1
                if label[vv] == label[i]:
                    f.write('{} {}\n'.format(i, vv))
    f.close()
    print('+: {}'.format(counter / (num * topk)))

'''
f = h5py.File('data/usps.h5', 'r')
train = f.get('train')
test = f.get('test')
X_tr = train.get('data')[:]
y_tr = train.get('target')[:]
X_te = test.get('data')[:]
y_te = test.get('target')[:]
f.close()
usps = np.concatenate((X_tr, X_te)).astype(np.float32)
label = np.concatenate((y_tr, y_te)).astype(np.int32)
'''

'''
hhar = np.loadtxt('data/hhar.txt', dtype=float)
label = np.loadtxt('data/hhar_label.txt', dtype=int)
'''

usps = np.loadtxt('data/hhar.txt', dtype=float)
usps_label = np.loadtxt('data/hhar_label.txt', dtype=int)

# official_stl10h = np.loadtxt('data3/official_stl10h.txt',dtype=float)
# official_stl10h_label = np.loadtxt('data3/official_stl10h_label.txt',dtype=float)

# construct_graph(reut, label, 'ncos')
if __name__ == "__main__":
    print('这是SDCN，不是trident')
    print()
    # construct_graph(reut, label, 'ncos')
    construct_graph(usps, usps_label, 'heat')