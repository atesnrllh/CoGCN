import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from utils import accuracy
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

class NegativeSamplingLoss(nn.Module):
    node_pair_list: list
    node_freq_list: list
    neg_sample_num: int
    Q: float

    def __init__(self, node_pair_list, neg_freq_list, neg_num=20, Q=10):
        super(NegativeSamplingLoss, self).__init__()
        self.node_pair_list = node_pair_list
        self.neg_freq_list = neg_freq_list
        self.neg_sample_num = neg_num
        self.Q = Q

    def forward(self, input_list):
        assert len(input_list) == 2
        node_embedding, batch_indices = input_list[0], input_list[1]
        node_embedding = [node_embedding] if not isinstance(node_embedding, list) and len(node_embedding.size()) == 2 else node_embedding
        return self.__negative_sampling_loss(node_embedding, batch_indices)

    # Negative sampling loss used for unsupervised learning to preserve local connective proximity
    def __negative_sampling_loss(self, node_embedding, batch_indices):
        bce_loss = nn.BCEWithLogitsLoss()
        neighbor_loss = Variable(torch.tensor([0.], device=batch_indices.device), requires_grad=True)
        timestamp_num = len(node_embedding)
        # print('timestamp num: ', timestamp_num)
        for i in range(timestamp_num):
            embedding_mat = node_embedding[i]   # tensor
            node_pairs = self.node_pair_list[i]  # list
            # print('node pairs: ', len(node_pairs[0]))
            node_freqs = self.neg_freq_list[i]  # tensor
            sample_num, node_indices, pos_indices, neg_indices = self.__get_node_indices(batch_indices, node_pairs, node_freqs)
            if sample_num == 0:
                continue
            # For this calculation block, we refer to some implementation details in https://github.com/aravindsankar28/DySAT/blob/master/models/DySAT/models.py
            # or https://github.com/kefirski/pytorch_NEG_loss/blob/master/NEG_loss/neg.py, or https://github.com/williamleif/GraphSAGE/blob/master/graphsage/models.py
            # or https://github.com/Andras7/word2vec-pytorch/blob/master/word2vec/model.py.
            # Here when calculating the neg_score, we use the 'matmul' operation. We can also use the 'mul' operation to calculate neg_score just like calculating pos_score
            # and using the 'mul' operation can reduce the computation complexity of neg_score calculation.
            pos_score = torch.sum(embedding_mat[node_indices].mul(embedding_mat[pos_indices]), dim=1)
            neg_score = torch.sum(embedding_mat[node_indices].matmul(torch.transpose(embedding_mat[neg_indices], 1, 0)), dim=1)
            # print('pos score: ', pos_score.mean().item(), 'pos max: ', pos_score.max().item(), 'pos min: ', pos_score.min().item())
            # print('neg score: ', neg_score.mean().item(), 'neg max: ', neg_score.max().item(), 'neg min: ', neg_score.min().item())
            pos_loss = bce_loss(pos_score, torch.ones_like(pos_score))
            neg_loss = bce_loss(neg_score, torch.zeros_like(neg_score))
            loss_val = pos_loss + self.Q * neg_loss
            neighbor_loss = neighbor_loss + loss_val
            # print('neighbor loss: ', neighbor_loss.item())
            ######################
        return neighbor_loss

    def __get_node_indices(self, batch_indices, node_pairs: np.ndarray, node_freqs: np.ndarray):
        device = batch_indices.device
        dtype = batch_indices.dtype
        node_indices, pos_indices, neg_indices = [], [], []
        random.seed()

        sample_num = 0
        for node_idx in batch_indices:
            # print('node pair type: ', type(node_pairs))
            neighbor_num = len(node_pairs[node_idx])
            if neighbor_num <= self.neg_sample_num:
                pos_indices += node_pairs[node_idx]
                real_num = neighbor_num
            else:
                pos_indices += random.sample(node_pairs[node_idx], self.neg_sample_num)
                real_num = self.neg_sample_num
            node_indices += [node_idx] * real_num
            sample_num += real_num
        if sample_num == 0:
            return sample_num, None, None, None
        neg_indices += random.sample(node_freqs, self.neg_sample_num)

        node_indices = torch.tensor(node_indices, dtype=dtype, device=device)
        pos_indices = torch.tensor(pos_indices, dtype=dtype, device=device)
        neg_indices = torch.tensor(neg_indices, dtype=dtype, device=device)
        return sample_num, node_indices, pos_indices, neg_indices




import scipy.sparse as sp
def get_node_pair_list(walk_file_path):

    node_pair_list = []
    walk_spadj = sp.load_npz(walk_file_path)
    neighbor_arr = walk_spadj.tolil().rows
    node_pair_list.append(neighbor_arr)

    return node_pair_list

import json
import os
def get_node_freq_list(freq_file_path):

    node_freq_list = []
    with open(freq_file_path, 'r') as fp:
        node_freq_arr = json.load(fp)
        node_freq_list.append(node_freq_arr)

    return node_freq_list

def get_loss():
    walk_pair_base_path = "gcn_walk_pairs/f_name.npz"
    node_freq_base_path = "gcn_node_freq/f_name.json"
    node_pair_list = get_node_pair_list(walk_pair_base_path)
    neg_freq_list  = get_node_freq_list(node_freq_base_path)

    loss = NegativeSamplingLoss(node_pair_list, neg_freq_list, neg_num=20, Q=20)

    return loss
