import numpy as np
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize


topk = 5


def construct_graph(features, label, method='cos'):
    fname = 'graph_ensemble/BC-5_graph.txt'
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


    inds = []
    for i in range(dist.shape[0]):

        ind = np.argpartition(dist[i, :], -(topk+1))[-(topk+1):]
        inds.append(ind)
        f = open(fname, 'w')
    counter = 0


    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                pass
            else:
                if label[vv] != label[i]:
                    counter += 1
                f.write('{} {}\n'.format(i, vv))
    f.close()
    print('+: {}'.format(counter / (num * topk)))

BC = np.loadtxt('data_ensemble/BC-.txt', dtype=float)
BC_label = np.loadtxt('data_ensemble/BC-_label.txt', dtype=float)

print(BC.shape)



if __name__ == "__main__":
    print('fsfdsfds')
    print()
    # construct_graph(reut_kmeans, label, 'ncos' 'heat')
    construct_graph(BC, BC_label, 'heat')
