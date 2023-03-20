# coding: utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttn(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttn, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc2 = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)

    def forward(self, hidden_states):
        # (batch_size, time_steps, hidden_size)
        score_first_part = self.fc1(hidden_states)
        # (batch_size, hidden_size)
        h_t = hidden_states[:,-1,:]
        # (batch_size, time_steps)
        score = torch.bmm(score_first_part, h_t.unsqueeze(2)).squeeze(2)
        attention_weights = F.softmax(score, dim=1)
        # (batch_size, hidden_size)
        context_vector = torch.bmm(hidden_states.permute(0,2,1), attention_weights.unsqueeze(2)).squeeze(2)
        # (batch_size, hidden_size*2)
        pre_activation = torch.cat((context_vector, h_t), dim=1)
        # (batch_size, hidden_size)
        attention_vector = self.fc2(pre_activation)
        attention_vector = torch.tanh(attention_vector)

        return attention_vector, attention_weights

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]
# K-core subgraph based diffusion layer
from layers import *
class CoreDiffusionx(nn.Module):
    input_dim: int
    output_dim: int
    layer_num: int
    bias: bool
    rnn_type: str

    def __init__(self, input_dim, output_dim, core_num=1, bias=True, rnn_type='GRU'):
        super(CoreDiffusionx, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.core_num = core_num
        self.rnn_type = rnn_type

        # self.linear = nn.Linear(input_dim, output_dim)
        # self.att_weight = nn.Parameter(torch.FloatTensor(core_num))
        assert self.rnn_type in ['LSTM', 'GRU']
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=output_dim, num_layers=1)
        else:
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=output_dim, num_layers=1)
        self.norm = nn.LayerNorm(output_dim)
        # self.reset_parameters()
        self.attention  = Attention(output_dim)
        # self.attn = TemporalAttn(hidden_size=256)

    # def reset_parameters(self):
    #     # stdv = 1. / math.sqrt(self.weight.size(1))
    #     self.att_weight.data.uniform_(0, 1)

    def forward(self, x, adj_list):
        hx_list = []
        # output = None
        for i, adj in enumerate(adj_list):
            if i == 2:

                x = torch.sparse.mm(adj, x)
                # res = hx_list[-1] + torch.sparse.mm(adj, x)
            # hx = self.linear(res)
            hx_list.append(x)
        hx_list = [F.relu(res) for res in hx_list]

        #################################
        # Simple Core Diffusion, no RNN
        # out = hx_list[0]
        # for i, res in enumerate(hx_list[1:]):
        #     out = out + res
        # output = self.linear(out)
        ##################################
        # Add RNN to improve performance, but this will reduce the computation efficiency a little.
        hx = torch.stack(hx_list, dim=0).transpose(0, 1)  # [batch_size, core_num, input_dim]
        outputs, hidden = self.rnn(hx)
        # x, weights = self.attn(hidden[-1])

        attn_weights = self.attention(hidden[-1], outputs)
        # context = attn_weights.bmm(outputs.transpose(0, 1))  # (B,1,N)
        context = torch.bmm(attn_weights.permute(2,1,0),outputs)
        context = context.squeeze(1)  # (1,B,N)

        # context = context.sum(dim=0)
        output = context
        # Layer normalization could improve performance and make rnn stable
        output = self.norm(output)
        return output


# K-core subgraph based diffusion layer
class CoreDiffusion(nn.Module):
    input_dim: int
    output_dim: int
    layer_num: int
    bias: bool
    rnn_type: str

    def __init__(self, input_dim, output_dim, core_num=1, bias=True, rnn_type='GRU'):
        super(CoreDiffusion, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.core_num = core_num
        self.rnn_type = rnn_type

        # self.linear = nn.Linear(input_dim, output_dim)
        # self.att_weight = nn.Parameter(torch.FloatTensor(core_num))
        assert self.rnn_type in ['LSTM', 'GRU']
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=output_dim, num_layers=1, bias=bias, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=output_dim, num_layers=1, bias=bias, batch_first=True)
        self.norm = nn.LayerNorm(output_dim)
        # self.reset_parameters()

    # def reset_parameters(self):
    #     # stdv = 1. / math.sqrt(self.weight.size(1))
    #     self.att_weight.data.uniform_(0, 1)

    def forward(self, x, adj_list):
        hx_list = []
        # output = None
        for i, adj in enumerate(adj_list):
            if i == 0:
                x1 = torch.sparse.mm(adj_list[0], x)
            # elif i == 1:
            #     res = torch.sparse.mm(adj_list[1], x)
                x = x1
            elif i == 1:
                x2 = torch.sparse.mm(adj_list[1], x)
                x = x2
            else:
                x3 =  torch.sparse.mm(adj_list[2], x1) # adj_list[1]
                x = x3
                #res = torch.sparse.mm(adj, x)
            # hx = self.linear(res)
            hx_list.append(x)
        hx_list = [F.relu(res) for res in hx_list]

        #################################
        # Simple Core Diffusion, no RNN
        # out = hx_list[0]
        # for i, res in enumerate(hx_list[1:]):
        #     out = out + res
        # output = self.linear(out)
        ##################################
        # Add RNN to improve performance, but this will reduce the computation efficiency a little.
        hx = torch.stack(hx_list, dim=0).transpose(0, 1)  # [batch_size, core_num, input_dim]
        output, _ = self.rnn(hx)
        output = output.sum(dim=1)
        # Layer normalization could improve performance and make rnn stable
        output = self.norm(output)
        return output


# Multi-Layer Perceptron(MLP) layer
class MLP(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int
    layer_num: int
    bias: bool
    activate_type: str

    def __init__(self, input_dim, hidden_dim, output_dim, layer_num, bias=True, activate_type='N'):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer_num = layer_num
        self.bias = bias
        self.activate_type = activate_type
        assert self.activate_type in ['L', 'N']
        assert self.layer_num > 0

        # if layer_num == 1:
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        # else:
        #     # self.linears = torch.nn.ModuleList()
        #     self.linears.append(nn.Linear(input_dim, hidden_dim, bias=bias))
        #     for layer in range(layer_num - 2):
        #         self.linears.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
        #     self.linears.append(nn.Linear(hidden_dim, output_dim, bias=bias))

    def forward(self, x):
        if self.layer_num == 1:  # Linear model
            x = self.linear(x)
            if self.activate_type == 'N':
                x = F.selu(x)
            # x = F.selu(x)
            return x
        # h = x  # MLP
        # for layer in range(self.layer_num):
        #     h = self.linears[layer](h)
        #     if self.activate_type == 'N':
        #         h = F.selu(h)
        # return h
