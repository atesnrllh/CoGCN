from __future__ import print_function, division
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils import load_data, load_graph
from GNN import GraphAttentionLayer
from torch_geometric.nn import GCNConv #GATConv
from evaluation import eva
import pickle
from models2 import CGCN
import utils
from pregae_pretrain import GAEx

# the graph autoencoder
class GAE(nn.Module):
    def __init__(self, num_features, hidden1_size, hidden2_size, embedding_size, alpha):
        super(GAE, self).__init__()
        self.embedding_size = embedding_size
        self.alpha = alpha
        # encoder configuration
        # self.conv11 = GCNConv(num_features, num_features)
        # self.conv12 = GCNConv(num_features, num_features)
        self.CGCN = CGCN(768, 256, 256, trans_num=1, diffusion_num=1, bias=True, rnn_type="GRU", model_type="C", trans_activate_type="L")
        self.CGCN.load_state_dict(torch.load(args.kcore_path, map_location='cpu'))
        # for param in self.CGCN.parameters():
        #     param.requires_grad = False


        self.cluster_layer = nn.Parameter(torch.Tensor(223, 256), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, adj_list, x, adj, M,edges_index):

        z = self.CGCN(adj_list, x)

        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / 1)
        q = q.pow((1 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return z, q


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred

import k_core_loss
def pretrain_gae(x, y, adj_path, userid, args, filename):
    model = GAE(num_features=args.input_dim, hidden1_size=args.hidden1_dim,  hidden2_size=args.hidden2_dim,
                  embedding_size=args.hidden3_dim, alpha=args.alpha).to(device)
    model = model.cuda()
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    adj, adj_label, edges_index, edges = load_graph(adj_path, len(y), args.k, filename, "pregae")

    adj_dense = adj.to_dense()
    adj_numpy = adj_dense.data.cpu().numpy()
    t=10
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    M = torch.Tensor(M_numpy).cuda()

    adj = adj_dense.cuda()
    adj_label = adj_label.cuda()

    data = x.cuda() #data1['feature'].cuda() # np.loadtxt(x_path, dtype=float)
    y = y#data1['label'].cpu().detach().numpy()# np.loadtxt(y_path, dtype=int)


    # utils.generate_k_core(edges)
    adj_list = utils.get_core_adj_list()

    # x_list = utils.get_date_adj_list()


    # data = torch.Tensor(dataset.x).cuda()
    # y = dataset.y
    with torch.no_grad():
        z, q = model(adj_list, data, adj, M,edges_index)

    kmeans = KMeans(n_clusters=223, n_init=20) #276
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    acc, acc_un, nmi, ari, f1 = eva(y, y_pred,0)

    import logging
    logging.info('epoch %d acc %.4f nmi %.4f ari %.4f f1 %.4f', 0, acc_un, nmi, ari, f1)

    # data = torch.Tensor(dataset.x).cuda()
    # y = dataset.y
    best_loss = float('inf')
    acc_best = -1
    model.train()

    loss_model = k_core_loss.get_loss()
    loss_model = loss_model.to("cuda:0")

    for epoch in range(args.epochs):


        #with torch.no_grad():
        z, tmp_q = model(adj_list, data, adj, M,edges_index)

        tmp_q = tmp_q.data
        p = target_distribution(tmp_q)

        res1 = tmp_q.cpu().numpy().argmax(1)  # Q
        res2 = p.data.cpu().numpy().argmax(1)  # P

        acc, acc_un, nmi, ari, f1 = eva(y, res2,0)
        logging.info('epoch %s acc %.4f nmi %.4f ari %.4f f1 %.4f', str(epoch) + ' P', acc_un, nmi, ari, f1)
        acc, acc_un, nmi, ari, f1 = eva(y, res1,0)
        logging.info('epoch %s acc %.4f nmi %.4f ari %.4f f1 %.4f', str(epoch) + ' Q', acc_un, nmi, ari, f1)
        # if acc > acc_best:
        #     acc_best, nmi_best, ari_best, f1_best = acc, nmi, ari, f1
        #     epoch_best = epoch

        # z, tmp_q = model(adj_list, data, adj, M,edges_index)
        #re_loss_gae = F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1))

        z, tmp_q = model(adj_list, data, adj, M,edges_index)

        kl_loss = F.kl_div(tmp_q.log(), p, reduction='batchmean')
        all_nodes = torch.arange(1424, device="cuda:0")
        node_indices = all_nodes[torch.randperm(1424)]

        loss_input_list = [z, node_indices]
        kcore_loss = loss_model(loss_input_list)

        loss =   1*kl_loss + 1*kcore_loss
        print(epoch,loss)
        #loss.requires_grad = True

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()


    f = open("acc", "a")
    f.write(str(acc)+"  "+str(acc_un)+"  "+str(nmi)+"  "+str(ari)+"  "+str(f1) + "\n")
    f.close()



import pre_kcore
if __name__ == "__main__":

    for i in range(10):
        filename = str(9-i)

        parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--name', type=str, default='qtmt')
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--k', type=int, default=5)
        parser.add_argument('--n_clusters', default=6, type=int)
        parser.add_argument('--hidden1_dim', default=256, type=int)
        parser.add_argument('--hidden2_dim', default=128, type=int)
        parser.add_argument('--hidden3_dim', default=16, type=int)
        parser.add_argument('--weight_decay', type=int, default=5e-3)
        parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')

        args = parser.parse_args()
        args.cuda = torch.cuda.is_available()
        print("use cuda: {}".format(args.cuda))
        device = torch.device("cuda" if args.cuda else "cpu")

        with open('/media/nuri/E/datasets/cross-session task extraction (CSTE)/lugo/UE/cste_feature.pickle', 'rb') as handle:
            x = pickle.load(handle)
            x= x.astype(np.float32)
        # with open('/home/nuri/Desktop/search-master/test', 'rb') as handle:
        #     x = pickle.load(handle)
        #     x= x.astype(np.float32)
        # from sklearn.decomposition import PCA
        # pca = PCA(n_components=256)
        # principalComponents = pca.fit_transform(x)
        # x = principalComponents.data
        x = torch.from_numpy(np.asarray(x)).cuda()
        # filename = "9"
        pre_kcore.func_pre_kcore(filename)
        # adj_path = "/media/nuri/E/datasets/QTMT/adj_list/"+filename+".txt" #data['label'].cpu().detach().numpy()# np.loadtxt(y_path, dtype=int)
        # #adj_path = "/media/nuri/E/datasets/QTMT/adj_list/"+filename+".txt"
        #
        # userid = np.loadtxt("/home/nuri/Downloads/DGC-EFR-main/DGC-EFR-tasking/graph/cste/agnobert/userid.txt", dtype=np.int64)
        # y = np.loadtxt("/media/nuri/E/datasets/QTMT/qtmt_label.txt", dtype=np.int64)

        adj_path = "/media/nuri/E/datasets/cross-session task extraction (CSTE)/lugo/UE/adj2/"+filename+".txt"

        userid = np.loadtxt("/home/nuri/Downloads/DGC-EFR-main/DGC-EFR-tasking/graph/cste/agnobert/userid.txt", dtype=np.int64)
        y = np.loadtxt("/media/nuri/E/datasets/cross-session task extraction (CSTE)/lugo/cste_label.txt", dtype=np.int64)

    #0 :acc 0.9627 :acc_un 0.5358 , nmi 0.8352 , ari 0.5213 , f1 0.5407 , f6 0.5310 english
    #0 :acc 0.9725 :acc_un 0.5611 , nmi 0.8191 , ari 0.5928 , f1 0.6068 , f6 0.6422 turkish
    #0 :acc 0.9700 :acc_un 0.5414 , nmi 0.8165 , ari 0.5470 , f1 0.5623 , f6 0.6011 french
    #0 :acc 0.9611 :acc_un 0.5183 , nmi 0.8263 , ari 0.5203 , f1 0.5405 , f6 0.5216 german
    #0 :acc 0.9619 :acc_un 0.5232 , nmi 0.8308 , ari 0.5268 , f1 0.5465 , f6 0.5289 Chinese PRC

        if args.name == 'acm':
            args.k = None
            args.n_clusters = 3
            args.input_dim = 1870

        if args.name == 'dblp':
            args.k = None
            args.n_clusters = 1299
            args.input_dim = 768

        if args.name == 'cora':
            args.k = None
            args.n_clusters = 225
            args.input_dim = 768

        if args.name == 'qtmt':
            args.k = None
            args.n_clusters = len(list(set(y)))
            args.input_dim = x.shape[1]

        args.kcore_path = 'data/kcore_{}.pkl'.format(args.name)
        args.gae_path = 'data/pregae_{}.pkl'.format(args.name)

        print(args)
        pretrain_gae(x,y,adj_path,userid, args, filename)
