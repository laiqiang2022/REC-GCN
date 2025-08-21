from __future__ import print_function, division
import argparse
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
from utils import load_data, load_graph
from GNN import GNNLayer
from evaluation import eva


class AE(nn.Module):# autoencoder

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)  #encoder
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))         # decoder
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z


class REC_GCN(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, 
                n_input, n_z, n_clusters, v=1):
        super(REC_GCN, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):

        x_bar, tra1, tra2, tra3, z = self.ae(x)
        
        sigma = 0.5

        h1 = self.gnn_1(x, adj)
        h2 = self.gnn_2((1 - sigma) * h1 + sigma * tra1, adj)
        h3 = self.gnn_3((1 - sigma) * h2 + sigma * tra2, adj)
        h4 = self.gnn_4((1 - sigma) * h3 + sigma * tra3, adj)
        h5 = self.gnn_5((1 - sigma) * h4 + sigma * z, adj, active=False)
        predict1 = F.softmax(h1, dim=1)
        predict2 = F.softmax(h2, dim=1)
        predict3 = F.softmax(h3, dim=1)
        predict4 = F.softmax(h4, dim=1)
        predict5 = F.softmax(h5, dim=1)


        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)

        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict1,predict2,predict3,predict4,predict5, z



def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_REC_GCN(dataset):
    model = REC_GCN(500, 500, 2000, 2000, 500, 500,
                n_input = args.n_input,
                n_z = args.n_z,
                n_clusters = args.n_clusters,
                v = 1.0).to(device)
    print(model)


    optimizer = Adam(model.parameters(), lr=args.lr)


    adj = load_graph(args.name,5)
    adj = adj.cuda()


    data = torch.Tensor(dataset.x).to(device)

    y = dataset.y
    with torch.no_grad():
        _, _, _,_, z = model.ae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 'pae')

    fname = 'result/{}_NMI_K=5(sigma0.5,0.1kl,0.1ce,lr=1e-3,epoch200,max_epo,nmi=).txt'.format(args.name)


    f = open(fname, 'w')
    epoch_list = []

    ACCZ, F1Z, NMIZ, ARIZ, PURZ, epoch_maxZ = 0, 0, 0, 0, 0, 0

    Epoch =200
    for epoch in range(Epoch):
        if epoch <Epoch:
            _, tmp_q, pred1, pred2, pred3, pred4, pred, _ = model(data, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            epoch_list.append(epoch)
            res1 = tmp_q.cpu().numpy().argmax(1)


            res2 = pred.data.cpu().numpy().argmax(1)
            res3 = p.data.cpu().numpy().argmax(1)
            a1, g1, acc_q, f1_q, nmi_q, ari_q, pur_q = eva(y, res1, str(epoch) + 'Q')
            a2, g2, acc_z, f1_z, nmi_z, ari_z, pur_z = eva(y, res2, str(epoch) + 'Z')
            a3, g3, acc_p, f1_p, nmi_p, ari_p, pur_p = eva(y, res3, str(epoch) + 'P')

            f.write(a1)
            f.write('\n')
            f.write(a2)
            f.write('\n')
            f.write(a3)
            f.write('\n')

            if nmi_z > NMIZ:
                NMIZ = nmi_z
                ACCZ = acc_z
                epoch_maxZ = epoch
                F1Z = f1_z
                ARIZ = ari_z
                PURZ = pur_z


        x_bar, q, _, _,_,_,pred, _ = model(data, adj)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)

        loss = 0.1* kl_loss + 0.1* ce_loss + re_loss


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    x2 = str(epoch_maxZ)
    b2 = str(':acc {:.4f}'.format(ACCZ))
    c2 = str(', nmi {:.4f}'.format(NMIZ))
    d2 = str(', ari {:.4f}'.format(ARIZ))
    e2 = str(', f1 {:.4f}'.format(F1Z))
    f2 = str(', pur {:.4f}'.format(PURZ))
    a4 = str("max epoch" + x2 + b2 + c2 + d2 + e2 + f2)
    f.write(a4)
    f.write('\n')

    f.close()

    return ACCZ, F1Z, NMIZ, ARIZ, PURZ, epoch_maxZ



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='BC-')
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_clusters', default=11, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--pretrain_path', type=str, default='pkl')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")


    dataset = load_data(args.name)
    args.n_input = dataset.x.shape[1]

    if args.name == 'BC-':
        args.n_clusters = 2
        args.n_z = 10
        args.pretrain_path = 'data_ensemble/BC.pkl'.format(args.name)


    print(args)
    ACCZ, F1Z, NMIZ, ARIZ, PURZ, epoch_maxZ =train_REC_GCN(dataset)
    print('Epoch: {}Z'.format(epoch_maxZ))
    print('ACC:{:.4f}'.format(ACCZ))
    print('NMI:{:.4f}'.format(NMIZ))
    print('ARI:{:.4f}'.format(ARIZ))
    print('F1:{:.4f}'.format(F1Z))
    print('PUR:{:.4f}'.format(PURZ))
