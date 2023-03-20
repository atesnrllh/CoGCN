import numpy as np
import scipy.sparse as sp
import h5py
import torch
import os
from torch.utils.data import Dataset
import shutil


def load_graph(adj_path,n, k, filename,train_type):
    # if k:
    #     path = 'graph/{}{}_graph.txt'.format(dataset, k)
    # else:
    #     path = 'graph/{}_graph.txt'.format(dataset)

    # data = torch.load("acm.pkl")
    # n, _ = data["feature"].shape
    edges_unordered = np.genfromtxt(adj_path, dtype=np.int32)
    #edges_unordered2 = np.genfromtxt("/home/nuri/Downloads/DGC-EFR-main/DGC-EFR-tasking_kcore_ses_ent_features/graph/cste_new/nonuserid/session/0.txt", dtype=np.int32) #0.25_4.txt"  0.21_6.txt
    edges_unordered2 = np.genfromtxt("/media/nuri/E/datasets/cross-session task extraction (CSTE)/lugo/session/session.txt", dtype=np.int32) #0.25_4.txt"  0.21_6.txt

    # if train_type == "pre_kcore":
    #     edges_unordered = np.unique(edges_unordered2, axis=0)
    if train_type == "pregae_pretrain" or train_type == "pregae" or train_type == "pre_kcore":
        #edges_unordered = np.concatenate((edges_unordered, edges_unordered2), axis=0)
        edges_unordered = np.unique(edges_unordered, axis=0)

    edges_unordered = np.unique(edges_unordered, axis=0)


    # edges_unordered2 = np.genfromtxt("/home/nuri/Downloads/DGC-EFR-main/DGC-EFR-tasking_kcore_features/graph/cste_new/nonuserid/session/0.txt", dtype=np.int32) #0.25_4.txt"
    #
    # edges_unordered = np.concatenate((edges_unordered, edges_unordered2), axis=0)
    #
    # edges_unordered = np.unique(edges_unordered, axis=0)
    # lxx = np.zeros(shape=(1423, 2))
    # for i in range(0, 1423):
    #     lxx[i][0] = int(i)
    #     lxx[i][1] = int(i+1)
    # edges_unordered = np.concatenate((lxx, edges_unordered), axis=0)
    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int64).reshape(edges_unordered.shape)
    #
    # txt_file = open("/home/nuri/Downloads/DGC-EFR-main/DGC-EFR-tasking/graph/cste/agnobert/nouserid/sim/"+str(filename)+".txt", "r")
    # file_content = txt_file.read()
    # content_list = file_content.split("\n")[:-1]
    # txt_file.close()

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)

    # adj = sp.coo_matrix((adj, (edges[:, 0], edges[:, 1])),
    #                     shape=(n, n), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_noeye = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj_noeye + sp.eye(adj_noeye.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj_label = sparse_mx_to_torch_sparse_tensor(adj_noeye)

    return adj,adj_label, torch.from_numpy(edges).T.cuda(), edges



def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class load_data(Dataset):
    def __init__(self, dataset):
        # self.x = np.loadtxt('data/{}.txt'.format(dataset), dtype=float)
        # self.y = np.loadtxt('data/{}_label.txt'.format(dataset), dtype=int)
        dataset = torch.load("acm.pkl")
        self.x = dataset['feature'] # np.loadtxt(x_path, dtype=float)
        self.y = dataset['label'].cpu().detach().numpy()# np.loadtxt(y_path, dtype=int)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])),\
               torch.from_numpy(np.array(self.y[idx])),\
               torch.from_numpy(np.array(idx))

def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)


def get_core_adj_list(max_core=-1):

    core_base_path = "gcn_cores"
    date_dir_list = sorted(os.listdir(core_base_path))
    core_adj_list = []

    # date_dir_path = os.path.join(core_base_path, date_dir_list)
    f_list = sorted(os.listdir(core_base_path))
    core_file_num = len(f_list)
    tmp_adj_list = []
    if max_core == -1:
        max_core = core_file_num
    f_list = f_list[:max_core]  # select 1 core to max core
    f_list = f_list[::-1]  # reverse order, max core, (max - 1) core, ..., 1 core

    # get k-core adjacent matrices at the i-th timestamp
    spmat_list = []
    for j, f_name in enumerate(f_list):
        spmat = sp.load_npz(os.path.join(core_base_path, f_name))
        spmat_list.append(spmat)
        # if j == 0:
        spmat = spmat + sp.eye(spmat.shape[0])
        # else:
        #     delta = spmat - spmat_list[j - 1]    # reduce subsequent computation complexity and reduce memory cost!
        #     if delta.sum() == 0:  # reduce computation complexity and memory cost!
        #         continue
        # Normalization will reduce the self weight, hence affect its performance! So we omit normalization.
        sptensor = sparse_mx_to_torch_sparse_tensor(spmat)
        tmp_adj_list.append(sptensor.cuda())
    # print('time: ', i, 'core len: ', len(tmp_adj_list))
    # core_adj_list.append(tmp_adj_list)

    return tmp_adj_list

def get_core_adj_listx1(max_core=-1):

    core_base_path = "gcn_cores"
    date_dir_list = sorted(os.listdir(core_base_path))
    core_adj_list = []

    # date_dir_path = os.path.join(core_base_path, date_dir_list)
    f_list = sorted(os.listdir(core_base_path))
    core_file_num = len(f_list)
    tmp_adj_list = []
    if max_core == -1:
        max_core = core_file_num
    f_list = f_list[:max_core]  # select 1 core to max core
    f_list = f_list[::-1]  # reverse order, max core, (max - 1) core, ..., 1 core

    # get k-core adjacent matrices at the i-th timestamp
    spmat_list = []
    for j, f_name in enumerate(f_list):
        spmat = sp.load_npz(os.path.join(core_base_path, f_name))
        spmat_list.append(spmat)
        if j == 0:
            spmat = spmat + sp.eye(spmat.shape[0])
        else:
            delta = spmat - spmat_list[j - 1]    # reduce subsequent computation complexity and reduce memory cost!
            if delta.sum() == 0:  # reduce computation complexity and memory cost!
                continue
        # Normalization will reduce the self weight, hence affect its performance! So we omit normalization.
        coo = spmat.tocoo()
        print('Acoo',coo)

        row = torch.from_numpy(coo.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(coo.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        tmp_adj_list.append(edge_index.cuda())
    # print('time: ', i, 'core len: ', len(tmp_adj_list))
    # core_adj_list.append(tmp_adj_list)

    return tmp_adj_list


def get_date_adj_list(self, origin_base_path, sep='\t', normalize=False, row_norm=False, add_eye=False, data_type='tensor'):
    assert data_type in ['tensor', 'matrix']
    date_dir_list = sorted(os.listdir(origin_base_path))
    # print('adj list: ', date_dir_list)
    date_adj_list = []

    original_graph_path = os.path.join(origin_base_path, date_dir_list[0])
    spmat = get_sp_adj_mat(original_graph_path, self.full_node_list, sep=sep)
    # spmat = sp.coo_matrix((np.exp(alpha * spmat.data), (spmat.row, spmat.col)), shape=(self.node_num, self.node_num))
    if add_eye:
        spmat = spmat + sp.eye(spmat.shape[0])
    if normalize:
        spmat = get_normalized_adj(spmat, row_norm=row_norm)
    # data type
    if data_type == 'tensor':
        sptensor = sparse_mx_to_torch_sparse_tensor(spmat)
        date_adj_list.append(sptensor.cuda() if self.has_cuda else sptensor)
    else:  # data_type == matrix
        date_adj_list.append(spmat)
    # print(len(date_adj_list))
    return date_adj_list


# Generate a row-normalized adjacent matrix from a sparse matrix
# If add_eye=True, then the renormalization trick would be used.
# For the renormalization trick, please refer to the "Semi-supervised Classification with Graph Convolutional Networks" paper,
# The paper can be viewed in https://arxiv.org/abs/1609.02907
def get_normalized_adj(adj, row_norm=False):
    """Row-normalize sparse matrix"""
    rowsum = np.array(adj.sum(1))
    p = -1 if row_norm else -0.5

    def inv(x, p):
        if p >= 0:
            return np.power(x, p)
        if x == 0:
            return x
        if x < 0:
            raise ValueError('invalid value encountered in power, x is negative, p is negative!')
        return np.power(x, p)
    inv_func = np.vectorize(inv)
    r_inv = inv_func(rowsum, p).flatten()
    r_mat_inv = sp.diags(r_inv)
    adj = r_mat_inv.dot(adj)
    if not row_norm:
        adj = adj.dot(r_mat_inv)
    adj = adj.tocoo()
    return adj




# Get sparse.lil_matrix type adjacent matrix
# Note that if you want to use this function, please transform the sparse matrix type, i.e. sparse.coo_matrix, sparse.csc_matrix if needed!
def get_sp_adj_mat(edges, full_node_list, sep='\t'):
    node_num = len(full_node_list)
    node2idx_dict = dict(zip(full_node_list, np.arange(node_num)))
    A = sp.lil_matrix((node_num, node_num))

    # ignore header
    for line in edges:
        line_list = line
        col_num = len(line_list)
        if col_num == 2:
            from_node, to_node, weight = line_list[0], line_list[1], 1
        else:
            from_node, to_node, weight = line_list[0], line_list[1], float(line_list[2])
        from_id = node2idx_dict[from_node]
        to_id = node2idx_dict[to_node]
        # remove self-loop data
        if from_id == to_id:
            continue
        A[from_id, to_id] = weight
        A[to_id, from_id] = weight
    A = A.tocoo()
    return A


import networkx as nx
import scipy.sparse as sp
def generate_k_corex1(edges):
    graph = nx.Graph()
    graph.add_edges_from(edges,weight=1)
    full_node_list = np.arange(0, 1424, 1, dtype=int)
    graph.add_nodes_from(full_node_list)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    core_num_dict = nx.core_number(graph)
    print("unique core nums: ", len(np.unique(np.array(list(core_num_dict.values())))))
    max_core_num = max(list(core_num_dict.values()))
    print('max core num: ', max_core_num)
    for i in range(1, max_core_num + 1):
        k_core_graph = nx.k_core(graph, k=i, core_number=core_num_dict)
        # print(i,len(k_core_graph.edges),len(k_core_graph.nodes))
        k_core_graph.add_nodes_from(full_node_list)
        ###############################
        # This node_list is quit important, or it will change the graph adjacent matrix and cause bugs!!!
        A = nx.to_scipy_sparse_array(k_core_graph, nodelist=full_node_list)
        ###############################
        sp.save_npz(os.path.join("gcn_cores", str(i) + '.npz'), A)
    print()


def delete_files(self, path):
    files = glob.glob(path+'/*')
    for f in files:
        os.remove(f)

def get_format_str(cnt):
    max_bit = 0
    while cnt > 0:
        cnt //= 10
        max_bit += 1
    format_str = '{:0>' + str(max_bit) + 'd}'
    return format_str

def check_and_make_path(to_make):
    if to_make == '':
        return
    if not os.path.exists(to_make):
        os.makedirs(to_make)

def generate_k_core(edges):
        # G.add_edges_from([(1, 2), (1, 3),(1,4),(1,5),(2,4),(3,4),(2,5),(3,5)])
        # G.add_edges_from([(1, 2), (1, 3),(1,4),(2,3),(2,4),(3,4),(4,5),(5,6),(5,7),(6,7),(7,8),(8,9),(8,10)])
        delete_files("gcn_cores")
        output_dir = "gcn_cores"

        Gx = nx.Graph()
        Gx.add_edges_from(edges)
        full_node_list = np.arange(0, 1424, 1, dtype=int)
        Gx.add_nodes_from(full_node_list)

        # for i in range(1424):
        #     Gx.add_edge(i, i)

        format_str = get_format_str(1)
        A = nx.to_scipy_sparse_matrix(Gx, nodelist=full_node_list)
        signature = format_str.format(1)
        check_and_make_path(output_dir)
        sp.save_npz(os.path.join(output_dir, signature + '.npz'), A)

        # import matplotlib.pyplot as plt


        G = nx.Graph()
        G.add_edges_from(edges)
        edge_of_core = []

        core_num_dict = nx.core_number(G)
        max_core_numx = max(list(core_num_dict.values()))
        counter = max_core_numx
        edge_of_core2 = nx.Graph()

        for i in range(0,max_core_numx):

            core_num_dict = nx.core_number(G)
            if len(core_num_dict) == 0:
                break
            max_core_num = max(list(core_num_dict.values()))

            k_core_graph = nx.k_core(G, k=max_core_num, core_number=core_num_dict)
            # nx.draw(k_core_graph, with_labels = True)
            e = k_core_graph.nodes

            # nx.draw(G, with_labels = True)
            # plt.savefig(str(max_core_num)+".png")

            print(counter)
            edge_of_core.extend(list(k_core_graph.edges()))
            # edge_of_core2 = nx.compose(k_core_graph,edge_of_core2)

                # ned = k_core_graph.number_of_edges()
                # nnod = k_core_graph.number_of_nodes()
                # strong_degre_of_connected_component.append(ned/((nnod*(nnod-1))/2))
                # core_strong_degre_of_connected_component.append(ned/((nnod*(nnod-1))/2))
            counter = counter - 1
            G.remove_nodes_from(e)
        print()
        #
        # with open('data/uci/1.formatc/'+input_file, 'w') as f:
        #     f.write("from_id"+"\t"+"to_id"+"\t"+"weight")
        #     f.write('\n')
        #     for a,b,c in edge_of_core2.edges.data():
        #         f.write(str(a)+"\t"+str(b)+"\t"+str(c["weight"]))
        #         print(str(a)+"\t"+str(b)+"\t"+str(c["weight"]))
        #         counter += 1
        #         f.write('\n')
        #
        # with open('data/uci/1.formatnc/'+input_file, 'w') as f:
        #     f.write("from_id"+"\t"+"to_id"+"\t"+"weight")
        #     f.write('\n')
        #     for a,b,c in Gx1.edges.data():
        #         f.write(str(a)+"\t"+str(b)+"\t"+str(c["weight"]))
        #         print(str(a)+"\t"+str(b)+"\t"+str(c["weight"]))
        #         counter += 1
        #         f.write('\n')
        #         # if counter < len(core_degre_graph.adj):
        #         #     f.write('\n')
        # print()
        # return

        # X = edge_list_of_lists
        # Y = strong_degre_of_connected_component
        # edge_list_of_lists = [x for _, x in sorted(zip(Y, X))]

        # step_lenght = max(strong_degre_of_connected_component)/max_core_num
        #
        # # distance = 1 / max_core_num

        # counter = 1
        # for ind in range(len(edge_of_core)):
        Gxxx = nx.Graph()
        Gxxx.add_edges_from(edge_of_core)
        Gxxx.add_nodes_from(full_node_list)

        graphs = list(nx.connected_components(Gxxx))
        print("nuber of subgraphs",len(graphs))

        A = nx.to_scipy_sparse_matrix(Gxxx, nodelist=full_node_list)
        signature = format_str.format(3)
        sp.save_npz(os.path.join(output_dir, signature + '.npz'), A)


        G1 = nx.Graph()
        G1.add_edges_from(edges)
        # full_node_list = np.arange(0, 1424, 1, dtype=int)
        G1.add_nodes_from(full_node_list)
        G1.remove_edges_from(edge_of_core)

        A = nx.to_scipy_sparse_matrix(G1, nodelist=full_node_list)
        signature = format_str.format(2)
        sp.save_npz(os.path.join(output_dir, signature + '.npz'), A)












import time
import json
def random_walk(spadj, walk_dir_path, freq_dir_path, f_name, walk_length, walk_time, weighted):
    t1 = time.time()
    node_num = spadj.shape[0]
    walk_len = walk_length + 1

    spadj = spadj.tolil()
    node_neighbor_arr = spadj.rows
    node_weight_arr = spadj.data
    walk_spadj = sp.lil_matrix((node_num, node_num))
    node_freq_arr = np.zeros(node_num, dtype=int)

    weight_arr_dict = dict()
    # random walk
    for nidx in range(node_num):
        for iter in range(walk_time):
            walk = [nidx]
            cnt = 1
            while cnt < walk_len:
                cur = walk[-1]
                neighbor_list = node_neighbor_arr[cur]
                if len(neighbor_list) == 0:
                    break
                if cur not in weight_arr_dict:
                    weight_arr = np.array(node_weight_arr[cur])
                    weight_arr = weight_arr / weight_arr.sum()
                    weight_arr_dict[cur] = weight_arr
                else:
                    weight_arr = weight_arr_dict[cur]
                nxt_id = np.random.choice(neighbor_list, p=weight_arr) if weighted else np.random.choice(neighbor_list)
                walk.append(int(nxt_id))
                cnt += 1
            # count walk pair
            seq_len = len(walk)
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    if walk[i] == walk[j]:
                        continue
                    from_id, to_id = walk[i], walk[j]
                    walk_spadj[from_id, to_id] = 1
                    walk_spadj[to_id, from_id] = 1
                    node_freq_arr[from_id] += 1
                    node_freq_arr[to_id] += 1
    t2 = time.time()
    print('random walk time: ', t2 - t1, ' seconds!')

    tot_freq = node_freq_arr.sum()
    Z = 0.00001
    neg_node_list = []
    for nidx in range(node_num):
        rep_num = int(((node_freq_arr[nidx] / tot_freq)**0.75) / Z)
        neg_node_list += [nidx] * rep_num
    walk_file_path = os.path.join(freq_dir_path, f_name.split('.')[0] + '.json')
    with open(walk_file_path, 'w') as fp:
        json.dump(neg_node_list, fp)
    del neg_node_list, node_freq_arr
    t3 = time.time()
    print('node freq time: ', t3 - t2, ' seconds!')

    walk_file_path = os.path.join(walk_dir_path, f_name.split('.')[0] + '.npz')
    sp.save_npz(walk_file_path, walk_spadj.tocoo())
    t4 = time.time()
    print('walk pair time: ', t4 - t3, ' seconds!')

import random_walk as rw
def generate_k_walk(edges):

    full_node_list = np.arange(0, 1424, 1, dtype=int)
    spadj = get_sp_adj_mat(edges, full_node_list, sep='\t')

    rw.random_walk(spadj, "gcn_walk_pairs", "gcn_node_freq", "f_name", 5, 20, True)


import os
import glob
def delete_files(path):
    files = glob.glob(path+'/*')
    for f in files:
        os.remove(f)
