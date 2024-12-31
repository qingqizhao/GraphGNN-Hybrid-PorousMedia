import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GNNLayer(MessagePassing):
    def __init__(self, node_dims, edge_dims, output_dims):
        super().__init__(aggr='add')  # "Add" aggregation.
        self.W_msg = nn.Linear(node_dims + edge_dims, node_dims)
        self.W_apply = nn.Linear(node_dims * 2, output_dims)
        
    def forward(self, x, edge_index, edge_attr):
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value="mean")
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        norm = deg.pow(-0.5)
        norm[deg == 0] = 0
        edge_weight = norm[row] * norm[col]
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_weight=edge_weight)
        return out

    def message(self, x_j, edge_attr, edge_weight):
        msg = torch.cat([x_j, edge_attr], dim=1)
        return edge_weight.view(-1, 1) * F.relu(self.W_msg(msg))

    def update(self, aggr_out, x):
        aggr_out = torch.cat([x, aggr_out], dim=1)
        return F.relu(self.W_apply(aggr_out))

class GCN(nn.Module):
    def __init__(self, node_feats, edge_feats, hidden_feats1=256, hidden_feats2=256,
                 predictor_hidden_feats=128, n_tasks=1):
        super().__init__()
        self.conv1 = GNNLayer(node_feats, edge_feats, hidden_feats1)
        self.conv2 = GNNLayer(hidden_feats1, edge_feats, hidden_feats2)
        self.readout = global_mean_pool
        self.fc1_bn = nn.BatchNorm1d(hidden_feats2)
        self.predict1 = nn.Linear(hidden_feats2, predictor_hidden_feats)
        self.predict2 = nn.Linear(predictor_hidden_feats, n_tasks)
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device)
        if data.batch is not None:
            batch = data.batch.to(device)
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long).to(device)

        h = self.conv1(x, edge_index, edge_attr)
        # Optionally uncomment for a second layer:
        # h = self.conv2(h, edge_index, edge_attr)

        graph_feats = self.readout(h, batch)
        graph_feats = self.fc1_bn(graph_feats)
        graph_feats = F.relu(self.predict1(graph_feats))
        out = self.predict2(graph_feats)
        return out
