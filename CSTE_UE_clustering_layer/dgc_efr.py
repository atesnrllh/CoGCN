from __future__ import print_function, division
import argparse
import os
import sys
import time
import glob
import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
from utils import load_data, load_graph, create_exp_dir
from GNN import GraphAttentionLayer
from evaluation import eva
from torch_geometric.nn import GCNConv #GATConv

from models2 import CGCN
import utils
import pickle
# The basic autoencoder
class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3,  n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z,  v=1):
        super(AE, self).__init__()

        self.conv11 = GCNConv(n_input, n_input)
        self.conv12 = GCNConv(n_input, n_input)

        # encoder
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)
        # decoder
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

        # degree
        self.v = v

    def forward(self, x, edges_index):

        x = self.conv11(x, edges_index)
        x = x.relu()
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv12(x, edges_index)
        x = x.relu()
        # x = F.dropout(x, p=0.5, training=self.training)

        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z

# graph autoencoder
class GAE(nn.Module):
    def __init__(self, num_features, hidden1_size, hidden2_size, embedding_size, alpha):
        super(GAE, self).__init__()
        # encoder
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.conv1 = GraphAttentionLayer(num_features, hidden1_size, alpha)
        self.conv2 = GraphAttentionLayer(hidden1_size, hidden2_size, alpha)
        self.conv3 = GraphAttentionLayer(hidden2_size, embedding_size, alpha)

    def forward(self, x, adj, M, enc_feat1, enc_feat2, enc_feat3):
        sigma = 0.1
        h = self.conv1(x, adj, M)
        h = self.conv2((1-sigma)*h + sigma*enc_feat1, adj, M)
        h = self.conv3((1-sigma)*h + sigma*enc_feat2, adj, M)
        z = F.normalize((1 -sigma)*h + sigma*enc_feat3, p=2, dim=1)
        # decoder
        A_pred = dot_product_decode(z)
        return A_pred, z


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
    return A_pred

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

class DGCEFR(nn.Module):
    def __init__(self, num_features, hidden1_size, hidden2_size, embedding_size, alpha, num_clusters, n_z, v=1):
        super(DGCEFR, self).__init__()
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.num_clusters = num_clusters
        self.v = v

        self.CGCN = CGCN(768, 768, 768, trans_num=1, diffusion_num=2, bias=True, rnn_type="GRU", model_type="C", trans_activate_type="L")
        self.CGCN.load_state_dict(torch.load(args.kcore_path, map_location='cpu'))
        for param in self.CGCN.parameters():
            param.requires_grad = False
        # pre-trained ae
        self.pre_ae =  AE(hidden1_size, hidden2_size, embedding_size, embedding_size, hidden2_size, hidden1_size, num_features, n_z = n_z)
        self.pre_ae.load_state_dict(torch.load(args.preae_path, map_location='cpu'))
        # pre-trained gae
        self.pre_gae = GAE(num_features, hidden1_size, hidden2_size, embedding_size, alpha)
        self.pre_gae.load_state_dict(torch.load(args.pregae_path, map_location='cpu'))

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(num_clusters, embedding_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)


    def forward(self, x_list,x, adj, M, edges_index):

        x = self.CGCN(x_list, x)

        x_bar, enc_h1, enc_h2, enc_h3, z = self.pre_ae(x,edges_index)
        A_pred, z = self.pre_gae(x, adj, M, enc_h1, enc_h2, enc_h3)

        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return A_pred, z, q, x_bar



def train(x,y,adj_path,adj_path2, userid, args, filename, filename2):

    # load initialized model
    model = DGCEFR(num_features=args.input_dim, hidden1_size=args.hidden1_dim, hidden2_size=args.hidden2_dim,
                  embedding_size=args.hidden3_dim, alpha=args.alpha, num_clusters=args.n_clusters, n_z=args.n_z ).to(device)


    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    # adjacent matrix
    adj, adj_label, edges_index, edges = load_graph(adj_path, len(y), args.k,filename)
    _, _, edges_index,_ = load_graph(adj_path2, len(y), args.k,filename2)

    # utils.generate_k_core(edges)

    adj_dense = adj.to_dense()
    adj_numpy = adj_dense.data.cpu().numpy()
    t =10
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    M = torch.Tensor(M_numpy).cuda()

    adj = adj_dense.cuda()
    adj_label = adj_label.cuda()

    # cluster parameter initiate
    data = x
    y = y

    adj_list = utils.get_core_adj_list()
    # x_list = utils.get_date_adj_list()

    with torch.no_grad():
         _, z,_, _ = model(adj_list, data, adj, M, edges_index)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    acc, nmi, ari, f1 = eva(y, y_pred, userid, 'pae')
    logging.info('epoch %d acc %.4f nmi %.4f ari %.4f f1 %.4f', 0, acc, nmi, ari, f1)
    acc_best, nmi_best, ari_best, f1_best, epoch_best = 0.0, 0.0, 0.0, 0.0, 0
    model.train()
    for epoch in range(args.epochs):

        if epoch % args.update_interval == 0:
            # update_interval
            A_pred, z, tmp_q, x_bar = model(adj_list, data, adj, M, edges_index)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            res1 = tmp_q.cpu().numpy().argmax(1)  # Q
            res2 = p.data.cpu().numpy().argmax(1)  # P

            acc, nmi, ari, f1 = eva(y, res2, userid, str(epoch) + 'P')
            logging.info('epoch %s acc %.4f nmi %.4f ari %.4f f1 %.4f', str(epoch) + ' P', acc, nmi, ari, f1)
            acc, nmi, ari, f1 = eva(y, res1, userid, str(epoch) + 'Q')
            logging.info('epoch %s acc %.4f nmi %.4f ari %.4f f1 %.4f', str(epoch) + ' Q', acc, nmi, ari, f1)
            if acc > acc_best:
                acc_best, nmi_best, ari_best, f1_best = acc, nmi, ari, f1
                epoch_best = epoch

        A_pred, z, q, x_bar = model(adj_list, data, adj, M, edges_index)

        kl_loss = F.kl_div(q.log(), p, reduction="batchmean")
        re_loss_gae = F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1))
        re_loss_ae = F.mse_loss(x_bar, data)
        # loss function with 3 parts
        loss =   kl_loss + re_loss_gae + re_loss_ae

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    print(epoch_best, ':acc_best {:.4f}'.format(acc_best), ', nmi {:.4f}'.format(nmi_best), ', ari {:.4f}'.format(ari_best),
            ', f1 {:.4f}'.format(f1_best))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='qtmt')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--k',type=int, default=5)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lambda1', type=float, default=1 )
    parser.add_argument('--n_clusters', default=6, type=int)
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--hidden1_dim', default=512, type=int)
    parser.add_argument('--hidden2_dim', default=256, type=int)
    parser.add_argument('--hidden3_dim', default=16, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--weight_decay', type=int, default=5e-3)
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
    parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
    parser.add_argument('--save', type=str, default='')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    args.save = './exp/eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    # load data
    # dataset = load_data(args.name)
    with open('/home/nuri/Downloads/DGC-EFR-main/DGC-EFR-tasking/graph/cste/agnobert/userid/cste_feature.pickle', 'rb') as handle:
        x = pickle.load(handle)
        x= x.astype(np.float32)
    x = torch.from_numpy(x).cuda()

    filename = "4/6"
    adj_path = "/home/nuri/Downloads/DGC-EFR-main/DGC-EFR-tasking/graph/cste/agnobert/nouserid_session_wiki/adj/"+filename+".txt" #data['label'].cpu().detach().numpy()# np.loadtxt(y_path, dtype=int)

    filename2 = "5" #5
    adj_path2 = "/home/nuri/Downloads/DGC-EFR-main/DGC-EFR-tasking/graph/cste/agnobert/nouserid/adj/"+filename2+".txt" #data['label'].cpu().detach().numpy()# np.loadtxt(y_path, dtype=int)


    userid = np.loadtxt("/home/nuri/Downloads/DGC-EFR-main/DGC-EFR-tasking/graph/cste/agnobert/userid.txt", dtype=np.int64)
    y = np.loadtxt("/home/nuri/Downloads/DGC-EFR-main/DGC-EFR-tasking/graph/cste/agnobert/userid/cste_label.txt", dtype=np.int64) #data['label'].cpu().detach().numpy()# np.loadtxt(y_path, dtype=int)


    # some configurations for datasets
    if args.name == 'acm':
        args.epochs = 200
        args.lr = 5e-3
        args.k = None
        args.n_clusters = 3
        args.input_dim = 1870

    if args.name == 'dblp':
        args.lr = 0.01
        args.epochs = 200
        args.k = None
        args.n_clusters = 4
        args.input_dim = 334

    if args.name == 'cora':
        args.epochs = 50
        args.lr = 0.001
        args.k = None
        args.n_clusters = 7
        args.input_dim = 1433

    if args.name == 'qtmt':
        args.k = None
        args.n_clusters = len(list(set(y)))
        args.input_dim = x.shape[1]

    # load pre-trained models
    args.pregae_path = 'data/pregae_{}.pkl'.format(args.name)
    args.preae_path = 'data/preae_{}.pkl'.format(args.name)
    args.kcore_path = 'data/kcore_{}.pkl'.format(args.name)
    print(args)
    train(x,y,adj_path,adj_path2, userid, args, filename, filename2)
