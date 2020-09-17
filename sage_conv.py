import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_scatter import scatter


def repr(m):
    return ' '.join(
        str([l.in_features, l.out_features]) for l in m.modules()
        if type(l) == nn.Linear)


def reset_parameters(m):
    for l in m.modules():
        if type(l) == nn.Linear:
            l.reset_parameters()


class SAGEConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv, self).__init__()

        self.lin_root = Linear(in_channels, out_channels, bias=False)
        self.lin_rel = Linear(in_channels, out_channels, bias=True)

        reset_parameters(self)

    def forward(self, x, res_size, edge_index):
        row, col = edge_index
        out = scatter(x[row], col, dim=0, dim_size=res_size, reduce="mean")
        out = self.lin_rel(out) + self.lin_root(x[:res_size])
        out = F.normalize(out, p=2, dim=-1)
        return out

    def __repr__(self):
        return repr(self)


# ===============================================================


class SAGEConvWithEdges(torch.nn.Module):
    def __init__(self, in_channels, in_edge_channels, out_channels):
        super(SAGEConvWithEdges, self).__init__()

        # self.edge_mlp = nn.Sequential(
        #     Linear(in_channels + in_edge_channels, in_channels),
        #     nn.ReLU(),
        #     # Linear(in_edge_channels, 1),
        # )
        # self.node_mlp_rel = Linear(in_channels, out_channels)
        self.node_mlp_rel = Linear(in_channels + in_edge_channels,
                                   out_channels)

        reset_parameters(self)

    def forward(self, x, res_size, edge_index, edge_attr):
        row, col = edge_index
        x_row = x[row]

        # edge_attr = x_row
        edge_attr = torch.cat([x_row, edge_attr], 1)

        # edge_attr = self.edge_mlp(edge_attr)
        # edge_attr = F.relu(x_row * self.edge_mlp(edge_attr).view(-1, 1))
        # edge_attr = F.relu(x_row + self.edge_mlp(edge_attr))
        edge_attr = F.normalize(edge_attr)

        x = scatter(edge_attr, col, dim=0, dim_size=res_size, reduce="mean")
        x = self.node_mlp_rel(x)
        x = F.normalize(x)
        return x

    def __repr__(self):
        return repr(self)
