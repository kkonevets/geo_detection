from __future__ import division
from tqdm import tqdm

import torch
from torch_cluster import neighbor_sampler
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.data.data import size_repr


class Block(object):
    def __init__(self, n_id, res_size, e_id, edge_index, size):
        self.n_id = n_id
        self.res_size = res_size
        self.e_id = e_id
        self.edge_index = edge_index
        self.size = size

    def __repr__(self):
        info = [(key, getattr(self, key)) for key in self.__dict__]
        info = ['{}={}'.format(key, size_repr(item)) for key, item in info]
        return '{}({})'.format(self.__class__.__name__, ', '.join(info))


class DataFlow(object):
    def __init__(self, n_id):
        self.n_id = n_id
        self.__last_n_id__ = n_id
        self.blocks = []

    @property
    def batch_size(self):
        return self.n_id.size(0)

    def append(self, n_id, res_size, e_id, edge_index):
        i, j = (1, 0)
        size = [None, None]
        size[i] = self.__last_n_id__.size(0)
        size[j] = n_id.size(0)
        block = Block(n_id, res_size, e_id, edge_index, tuple(size))
        self.blocks.append(block)
        self.__last_n_id__ = n_id

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        return self.blocks[::-1][idx]

    def __iter__(self):
        for block in self.blocks[::-1]:
            yield block

    def to(self, device):
        for block in self.blocks:
            block.edge_index = block.edge_index.to(device)
        return self

    def __repr__(self):
        n_ids = [self.n_id] + [block.n_id for block in self.blocks]
        sep = '<-'
        info = sep.join([str(n_id.size(0)) for n_id in n_ids])
        return '{}({})'.format(self.__class__.__name__, info)


class NeighborSampler(object):
    r"""The neighbor sampler from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper which iterates
    over graph nodes in a mini-batch fashion and constructs sampled subgraphs
    of size :obj:`num_hops`.

    It returns a generator of :obj:`DataFlow` that defines the message
    passing flow to the root nodes via a list of :obj:`num_hops` bipartite
    graph objects :obj:`edge_index` and the initial start nodes :obj:`n_id`.

    Args:
        data (torch_geometric.data.Data): The graph data object.
        size (int or float or [int] or [float]): The number of neighbors to
            sample (for each layer). The value of this parameter can be either
            set to be the same for each neighborhood or percentage-based.
        num_hops (int): The number of layers to sample.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
    """
    def __init__(self, config, data, deg=None, batch_size=1):
        self.nsample = config.nsample
        self.data = data
        self.num_hops = len(self.nsample)
        self.batch_size = batch_size
        self.shuffle = config.dev_mode
        self.edge_index = data.edge_index  # edge_index[0] -> edge_index[1]

        if deg is None:
            deg = degree(self.edge_index[1].to(torch.long),
                         data.num_nodes,
                         dtype=torch.int)
        self.cumdeg = torch.cat(
            [deg.new_zeros(1, dtype=torch.long),
             deg.cumsum(0)])

        self.tmp = torch.empty(data.num_nodes, dtype=torch.long)

    def __get_batches__(self, subset=None):
        r"""Returns a list of mini-batches from the initial nodes in
        :obj:`subset`."""

        if subset.dtype == torch.bool or subset.dtype == torch.uint8:
            subset = subset.nonzero().view(-1)
        if self.shuffle:
            subset = subset[torch.randperm(subset.size(0))]

        subsets = torch.split(subset, self.batch_size)
        assert len(subsets) > 0
        return subsets

    def __renumerate__(self, nodes, unique_nodes):
        self.tmp[unique_nodes] = torch.arange(unique_nodes.size(0))
        return self.tmp[nodes]

    def __produce_bipartite_data_flow__(self, n_id):
        r"""Produces a :obj:`DataFlow` object with a bipartite assignment
        matrix for a given mini-batch :obj:`n_id`."""

        data_flow = DataFlow(n_id)

        all_n_id = n_id
        for l in range(self.num_hops):
            e_id = neighbor_sampler(n_id, self.cumdeg, self.nsample[l])
            sub_edge_index = self.edge_index[:, e_id].to(torch.long)

            edges = [None, None]

            # ======================

            row_0 = torch.cat([sub_edge_index[0], all_n_id])
            row_1 = torch.cat([sub_edge_index[1], all_n_id])

            edges[1] = self.__renumerate__(row_1, all_n_id)

            n_id = sub_edge_index[0].unique(sorted=False)

            res_size = all_n_id.size(0)  # target nodes are placed first
            all_n_id = torch.cat([all_n_id, n_id])

            # res_size = all_n_id.size(0)
            # all_n_id = torch.cat([all_n_id, n_id])
            # all_n_id, inv = all_n_id.unique(sorted=False, return_inverse=True)
            # res_n_id = inv[:res_size]

            edges[0] = self.__renumerate__(row_0, all_n_id)

            # ======================

            edge_index = torch.stack(edges, dim=0)
            data_flow.append(all_n_id, res_size, e_id, edge_index)

        return data_flow

    def __call__(self, subset=None):
        r"""Returns a generator of :obj:`DataFlow` that iterates over the nodes
        in :obj:`subset` in a mini-batch fashion.

        Args:
            subset (LongTensor or BoolTensor, optional): The initial nodes to
                propagete messages to. If set to :obj:`None`, will iterate over
                all nodes in the graph. (default: :obj:`None`)
        """
        for n_id in self.__get_batches__(subset):
            yield self.__produce_bipartite_data_flow__(n_id)
