import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import Linear
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from evaluation import eva
import pickle

from torch_geometric.nn import GCNConv #GATConv
from utils import load_data, load_graph
# the basic autoencoder
class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3,  n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z,  v=1):
        super(AE, self).__init__()

        self.conv11 = GCNConv(n_input, n_input)
        self.conv12 = GCNConv(n_input, n_input)


        # encoder configuration
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)
        # decoder configuration
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)
        # degree
        self.v = v

    def forward(self, x, edges_index):
        x = self.conv11(x, edges_index)
        x = x.relu()
        # x = F.dropout(x, p=0.01, training=self.training)
        x = self.conv12(x, edges_index)
        x = x.relu()
        # x = F.dropout(x, p=0.01, training=self.training)


        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, z


class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001 * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# pre-train the autoencoder model
def pretrain_ae(model, dataset, args, y, userid, adj_path, filename):

    _, _, edges_index,_ = load_graph(adj_path, len(y), args.k, filename)

    train_loader = DataLoader(dataset, batch_size=1424, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    best_loss = float('inf')
    for epoch in range(args.epochs):
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.cuda()

            x_bar, _ = model(x,edges_index)
            loss = F.mse_loss(x_bar, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        with torch.no_grad():
            x = torch.Tensor(dataset.x).cuda().float()
            x_bar, z = model(x,edges_index)
            loss = F.mse_loss(x_bar, x)
            print('{} loss: {}'.format(epoch, loss))
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(z.data.cpu().numpy())
            eva(y, kmeans.labels_, userid, epoch)

            if best_loss > loss:
                best_loss = loss
                torch.save(model.state_dict(), 'data/preae_{}.pkl'.format(args.name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='qtmt')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--n_clusters', default=3, type=int)
    parser.add_argument('--n_z', default=128, type=int)
    parser.add_argument('--pretrain_path', type=str, default='pkl')
    parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
    parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    args.pretrain_path = 'data/{}.pkl'.format(args.name)

    #data = torch.load("acm.pkl")

    x_path = 'data/{}.txt'.format(args.name)
    y_path = 'data/{}_label.txt'.format(args.name)

    with open('/home/nuri/Downloads/DGC-EFR-main/DGC-EFR-tasking/graph/cste/agnobert/userid/cste_feature.pickle', 'rb') as handle:
        x = pickle.load(handle)
    x = torch.from_numpy(x)
    # x = torch.load("") #data['feature'] # np.loadtxt(x_path, dtype=float)
    y = np.loadtxt("/home/nuri/Downloads/DGC-EFR-main/DGC-EFR-tasking/graph/cste/agnobert/userid/cste_label.txt", dtype=np.int64) #data['label'].cpu().detach().numpy()# np.loadtxt(y_path, dtype=int)

    userid = np.loadtxt("/home/nuri/Downloads/DGC-EFR-main/DGC-EFR-tasking/graph/cste/agnobert/userid.txt", dtype=np.int64)

    filename = "6" #5
    adj_path = "/home/nuri/Downloads/DGC-EFR-main/DGC-EFR-tasking/graph/cste/agnobert/nouserid/adj/"+filename+".txt" #data['label'].cpu().detach().numpy()# np.loadtxt(y_path, dtype=int)
    # adj_path = "/home/nuri/Downloads/DGC-EFR-main/DGC-EFR-tasking/graph/cste/agnobert/nouserid/adj_squence.txt" #data['label'].cpu().detach().numpy()# np.loadtxt(y_path, dtype=int)

    if args.name == 'acm':
        args.k = None
        args.n_clusters = 3
        args.n_input = 1870

    if args.name == 'dblp':
        args.k = None
        args.n_clusters = 1299
        args.n_input = 768

    if args.name == 'cora':
        args.k = None
        args.n_clusters = 225
        args.n_input = 768

    if args.name == 'qtmt':
        args.k = None
        args.n_clusters = len(list(set(y)))
        args.n_input = x.shape[1]


    model = AE(
            n_enc_1=512,
            n_enc_2=256,
            n_enc_3=16,
            n_dec_1=16,
            n_dec_2=256,
            n_dec_3=512,
            n_input=args.n_input,
            n_z=10,).cuda()


    dataset = LoadDataset(x)
    pretrain_ae(model, dataset,args, y, userid, adj_path, filename)
