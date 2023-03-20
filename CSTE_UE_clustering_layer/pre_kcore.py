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


# the graph autoencoder
# class GAE(nn.Module):
#     def __init__(self, num_features, hidden1_size, hidden2_size, embedding_size, alpha):
#         super(GAE, self).__init__()
#         self.embedding_size = embedding_size
#         self.alpha = alpha
#
#         self.CGCN = CGCN(768, 768, 768, trans_num=1, diffusion_num=2, bias=True, rnn_type="GRU", model_type="C", trans_activate_type="L")
#
#     def forward(self, adj_list, x, adj, M,edges_index):
#
#         x = self.CGCN(adj_list, x)
#
#         return x


def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred

import k_core_loss
def pretrain_gae(x, y, adj_path, userid, args, filename):

    model = CGCN(768, 256, 256, trans_num=1, diffusion_num=1, bias=True, rnn_type="GRU", model_type="C", trans_activate_type="L").cuda()

    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    adj, adj_label, edges_index, edges = load_graph(adj_path, len(y), args.k, filename,"pre_kcore")

    adj_dense = adj.to_dense()
    adj_numpy = adj_dense.data.cpu().numpy()
    t=10
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    M = torch.Tensor(M_numpy).cuda()

    adj = adj_dense.cuda()
    adj_label = adj_label.cuda()

    data = x #data1['feature'].cuda() # np.loadtxt(x_path, dtype=float)
    y = y#data1['label'].cpu().detach().numpy()# np.loadtxt(y_path, dtype=int)

    utils.delete_files("gcn_cores")
    utils.delete_files("gcn_node_freq")
    utils.delete_files("gcn_walk_pairs")
    utils.generate_k_core(edges)
    utils.generate_k_walk(edges)
    print("cores are created")

    adj_list = utils.get_core_adj_list()

    # x_list = utils.get_date_adj_list()

    loss_model = k_core_loss.get_loss()
    loss_model = loss_model.to("cuda:0")

    # data = torch.Tensor(dataset.x).cuda()
    # y = dataset.y
    best_loss = float('inf')
    model.train()

    all_nodes = torch.arange(1424, device="cuda:0")
    node_indices = all_nodes[torch.randperm(1424)]
    best_loss = float('inf')
    for epoch in range(200):

        embedding_list = model(adj_list, data)
        loss_input_list = [embedding_list, node_indices]
        loss = loss_model(loss_input_list)
        print("epoch", epoch, "loss: ",loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if best_loss > loss:
            best_loss = loss
            torch.save(model.state_dict(), 'data/kcore_{}.pkl'.format(args.name))
            torch.save(embedding_list, 'core_qtmt.pt')


        # with torch.no_grad():
        #     embedding_list = model(adj_list, data)
        #     kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(z.data.cpu().numpy())
        #     acc, _,_,_ = eva(y, kmeans.labels_, userid, epoch)
        #     # print(loss)
        #     if best_loss > loss:
        #         best_loss = loss
        #         torch.save(model.state_dict(), 'data/pregae_{}.pkl'.format(args.name))

#if __name__ == "__main__":
def func_pre_kcore(filename):
    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='qtmt')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--k', type=int, default=5) 
    parser.add_argument('--n_clusters', default=6, type=int)
    parser.add_argument('--hidden1_dim', default=512, type=int)
    parser.add_argument('--hidden2_dim', default=256, type=int)
    parser.add_argument('--hidden3_dim', default=16, type=int)
    parser.add_argument('--weight_decay', type=int, default=5e-4)
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    # data = torch.load("acm.pkl")
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
    #filename = "5"
    # adj_path = "/media/nuri/E/datasets/QTMT/adj_list/"+filename+".txt" #data['label'].cpu().detach().numpy()# np.loadtxt(y_path, dtype=int)
    # #adj_path = "/media/nuri/E/datasets/QTMT/adj_list/"+filename+".txt"
    #
    # userid = np.loadtxt("/home/nuri/Downloads/DGC-EFR-main/DGC-EFR-tasking/graph/cste/agnobert/userid.txt", dtype=np.int64)
    # y = np.loadtxt("/media/nuri/E/datasets/QTMT/qtmt_label.txt", dtype=np.int64)

    adj_path = "/media/nuri/E/datasets/cross-session task extraction (CSTE)/lugo/UE/adj2/"+filename+".txt" #data['label'].cpu().detach().numpy()# np.loadtxt(y_path, dtype=int)
    #adj_path = "/media/nuri/E/datasets/QTMT/adj_list/"+filename+".txt"

    userid = np.loadtxt("/home/nuri/Downloads/DGC-EFR-main/DGC-EFR-tasking/graph/cste/agnobert/userid.txt", dtype=np.int64)
    y = np.loadtxt("/media/nuri/E/datasets/cross-session task extraction (CSTE)/lugo/cste_label.txt", dtype=np.int64)



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


    print(args)
    pretrain_gae(x,y,adj_path,userid, args, filename)
