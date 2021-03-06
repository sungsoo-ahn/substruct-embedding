import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.inits import zeros

NUM_ATOM_TYPES = 120
NUM_CHIRALITY_TAGS = 3

NUM_BOND_TYPES = 6
NUM_BOND_DIRECTIONS = 3

AGGR = "add"
NUM_LAYERS = 5


class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim), nn.ReLU(), nn.Linear(2 * emb_dim, emb_dim),
        )
        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPES, emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTIONS, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = AGGR

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]
        edge_index = edge_index.to(torch.long)

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class NodeEncoder(nn.Module):
    def __init__(self, num_layers, emb_dim, drop_rate):
        super(NodeEncoder, self).__init__()
        self.num_layers = num_layers
        self.emb_dim = emb_dim

        self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPES, emb_dim)
        self.x_embedding2 = nn.Embedding(NUM_CHIRALITY_TAGS, emb_dim)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.layers = nn.ModuleList([GINConv(emb_dim) for _ in range(num_layers)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layers)])
        self.relus = nn.ModuleList([nn.ReLU(emb_dim) for _ in range(num_layers - 1)])
        self.dropouts = nn.ModuleList([nn.Dropout(p=drop_rate) for _ in range(num_layers)])

    def forward(self, x, edge_index, edge_attr):
        out = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        for layer_idx in range(self.num_layers):
            out = self.layers[layer_idx](out, edge_index, edge_attr)
            out = self.batch_norms[layer_idx](out)
            if layer_idx < self.num_layers - 1:
                out = self.relus[layer_idx](out)

            out = self.dropouts[layer_idx](out)

        return out


class NodeEncoderWithHead(nn.Module):
    def __init__(self, num_head_layers, head_dim, num_encoder_layers, emb_dim, drop_rate):
        super(NodeEncoderWithHead, self).__init__()

        self.encoder = NodeEncoder(
            num_layers=num_encoder_layers, emb_dim=emb_dim, drop_rate=drop_rate
        )
        if num_head_layers == 1:
            self.head = nn.Linear(emb_dim, head_dim)
        elif num_head_layers == 2:
            self.head = nn.Sequential(
                nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, head_dim),
            )
        else:
            raise NotImplementedError

    def forward(self, x, edge_index, edge_attr):
        out = self.encoder(x, edge_index, edge_attr)
        out = self.head(out)

        return out


class GraphEncoder(NodeEncoder):
    def __init__(self, num_layers, emb_dim, drop_rate):
        super(GraphEncoder, self).__init__(num_layers, emb_dim, drop_rate=drop_rate)

    def forward(self, x, edge_index, edge_attr, batch):
        out = super(GraphEncoder, self).forward(x, edge_index, edge_attr)
        out = global_mean_pool(out, batch)

        return out


class GraphEncoderWithHead(nn.Module):
    def __init__(self, num_head_layers, head_dim, num_encoder_layers, emb_dim, drop_rate):
        super(GraphEncoderWithHead, self).__init__()

        self.encoder = GraphEncoder(
            num_layers=num_encoder_layers, emb_dim=emb_dim, drop_rate=drop_rate
        )
        if num_head_layers == 1:
            self.head = nn.Linear(emb_dim, head_dim)
        elif num_head_layers == 2:
            self.head = nn.Sequential(
                nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, head_dim),
            )
        else:
            raise NotImplementedError

    def forward(self, x, edge_index, edge_attr, batch):
        out = self.encoder(x, edge_index, edge_attr, batch)
        out = self.head(out)

        return out
