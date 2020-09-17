import torch
from torch.nn import Linear
from torch_geometric.utils import degree
from torch_geometric.data import Data
from sampler_geometric import NeighborSampler
from geometric import transductive_sage
from data import Config
import numpy as np
from sklearn.model_selection import train_test_split


def prepare_dataset(dataset, config, num_features):
    data = dataset.data
    zeros = torch.zeros(num_features, device=config.device)
    ones = torch.ones(num_features, device=config.device)
    stat = Data(mu_node=zeros, std_node=ones)
    data.stat = stat
    data.num_classes = data.y.unique().shape[0]
    ei = data.edge_index.numpy()[::-1].copy()
    data.edge_index = torch.from_numpy(ei)
    return data


def test_karate():
    from torch_geometric.datasets import KarateClub

    dataset = KarateClub('../data/KarateClub')
    config = Config('../data/KarateClub/', True, multilabel=False)
    data = prepare_dataset(dataset, config, 2)
    index = np.arange(data.y.shape[0])
    train, test = train_test_split(index,
                                   stratify=data.y,
                                   test_size=0.6,
                                   random_state=0)
    data.train_mask = torch.zeros(index.shape[0], dtype=torch.uint8)
    data.train_mask[train] = 1
    data.test_mask = torch.zeros(index.shape[0], dtype=torch.uint8)
    data.test_mask[test] = 1

    # ss = sparse.coo_matrix(data.x.numpy())
    # i = torch.LongTensor([ss.row, ss.col])
    # data.x = torch.sparse.FloatTensor(
    #     i, torch.from_numpy(ss.data), torch.Size(ss.shape))

    import networkx as nx

    g = nx.from_edgelist(data.edge_index.numpy().T)
    train_set = set(train)
    x = []
    for nid in sorted(g.nodes):
        inter = list(train_set.intersection(g.neighbors(nid)))
        ixs, vals = torch.unique(data.y[inter], return_counts=True)
        counts = torch.zeros(data.num_classes, dtype=torch.int64)
        counts[ixs] = vals
        x.append(counts)

    data.x = torch.stack(x).to(torch.float)
    # sub_edge_index, _ = subgraph(torch.from_numpy(train), data.edge_index)

    deg = degree(data.edge_index[1].to(torch.long),
                 data.num_nodes,
                 dtype=torch.int)
    loader = NeighborSampler(data,
                             size=[5, 5],
                             deg=deg,
                             batch_size=5,
                             shuffle=True)

    transductive_sage(loader, config, steps=200, test_every=20)


def test_pubmed():
    from torch_geometric.datasets import Planetoid

    dataset = Planetoid('../data/', 'PubMed')
    config = Config('../data/PubMed/', True, multilabel=False)
    data = prepare_dataset(dataset, config, dataset.data.num_features)

    print("train/val/test split: %i/%i/%i" %
          (data.train_mask.sum(), data.val_mask.sum(), data.test_mask.sum()))

    # data = Data(x=torch.ones(6, 1, dtype=torch.float32),
    #             edge_index=torch.LongTensor(
    #                 [[2, 2, 3, 4, 0, 1, 4, 2, 3, 5, 2, 4],
    #                  [0, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 5]]),
    #             y=torch.LongTensor([0, 0, 0, 1, 0, 0]))
    # data.num_classes = 2
    # data.train_mask = torch.ByteTensor([1, 1, 0, 1, 0, 1])
    # data.test_mask = torch.ByteTensor([0, 0, 1, 0, 1, 0])

    deg = degree(data.edge_index[1].to(torch.long),
                 data.num_nodes,
                 dtype=torch.int)
    loader = NeighborSampler(data,
                             size=[10, 10],
                             deg=deg,
                             batch_size=50,
                             shuffle=True)

    transductive_sage(loader, config, steps=200, test_every=20)


def check():
    def check_cities():
        num_nodes = len(load_vector("../data/nodes.bin", 'I'))
        vec = load_vector("../data/vk/neib_ftrs.bin", 'I')
        m = sparse.coo_matrix((vec[2::3], (vec[::3], vec[1::3])),
                              shape=(num_nodes, max(vec[1::3]) + 1),
                              dtype='float32').tocsr()
        cities = pd.read_csv("../data/vk/geography_splited.csv",
                             header=None)[0]

        ids = load_vector("../data/vk/nodes.bin", 'I')
        ix = 30
        row = m[ix].A[0]
        nz = np.nonzero(row)[0]
        res = [(c, i) for c, i in zip(cities[nz].values, row[nz])]
        sorted(res, key=lambda tup: tup[1])
        print(res)

    def base_line(data):
        x = data.x[np.where(data.test_mask == 1)[0]]
        if type(data.x) == sparse.csr_matrix:
            x = torch.from_numpy(x.A)
        pred = x.max(1)[1]
        correct = pred.eq(data.y[data.test_mask == 1]).sum().item()
        return correct / data.test_mask.sum().item()  # 0.725 - 100, 0.59 - 454

    def index_check():
        deg = torch.from_numpy(
            np.array(load_vector('../data/vk/degree.bin', 'I'), dtype=np.long))
        # small = ((deg < 10) & (deg > 0)).nonzero()

        ids = load_vector("../data/vk/nodes.bin", 'I')
        index = load_index()
        ix = 30
        nz = (index[1] == ix).nonzero()[:, 0]  # должны идти подряд !!!!
        print("friends of %d: " % ids[ix], [ids[i] for i in index[0][nz]])
        print(deg[ix])


if __name__ == "__main__":
    test_pubmed()
    test_karate()