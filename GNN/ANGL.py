import torch
from torch.nn import ModuleList, Linear
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm, PNAConv, RGCNConv, GATConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set

from encoder import NodeEncoder, EdgeEncoder

net_choices_available = ['sage', 'pna', 'rgcn', 'gat']


class GNN(torch.nn.Module):
    def __init__(self, num_tasks=1, num_layer=4, emb_dim=1433, drop_ratio=0.5, JK="sum", residual=False,
                 graph_pooling="sum", net_type="sage"):
        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        # add residual connection or not
        self.residual = residual
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.net_type = net_type

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_emb = NodeEncoder(emb_dim)
        self.edge_emb = EdgeEncoder(emb_dim=emb_dim)

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for i in range(num_layer):
            if self.net_type == 'sage':
                self.convs.append(SAGEConv(emb_dim, emb_dim))
            elif self.net_type == 'pna':
                self.convs.append(PNAConv(emb_dim, emb_dim, aggregators=['mean', 'min', 'max', 'std'],
                                          scalers=['identity', 'amplification'], deg=torch.tensor(1), towers=1,
                                          pre_layers=4, post_layers=4, divide_input=False))
            elif self.net_type == 'rgcn':
                self.convs.append(RGCNConv(emb_dim, emb_dim, num_relations=4, num_bases=30))
            elif self.net_type == 'gat':
                self.convs.append(GATConv(emb_dim, emb_dim, heads=1))
            else:
                raise ValueError(f"Invalid net type: {self.net_type}")

            self.batch_norms.append(BatchNorm(emb_dim))

        # Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                            torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        self.graph_pred_linear = ModuleList()
        self.graph_norm = ModuleList()

        if graph_pooling == "set2set":
            self.graph_pred_linear.append(Linear(2 * emb_dim, 2 * emb_dim))
            self.graph_pred_linear.append(Linear(2 * emb_dim, emb_dim))
            self.graph_pred_linear.append(Linear(emb_dim, self.num_tasks))

            self.graph_norm.append(BatchNorm(2 * emb_dim))
            self.graph_norm.append(BatchNorm(emb_dim))
        else:
            self.graph_pred_linear.append(Linear(emb_dim, 2 * emb_dim))
            self.graph_pred_linear.append(Linear(2 * emb_dim, emb_dim))
            self.graph_pred_linear.append(Linear(emb_dim, self.num_tasks))

            self.graph_norm.append(BatchNorm(2 * emb_dim))
            self.graph_norm.append(BatchNorm(emb_dim))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.att.reset_parameters()
        self.graph_pred_linear.reset_parameters()

    def forward(self, batched_data):
        # x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        x, edge_index = batched_data.x, batched_data.edge_index
        batch = None

        # h_list = [self.node_emb(x)]
        h_list = [x]
        # edge_embedding = self.edge_emb(edge_attr)
        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        # print("\n\n", f"forward1: {h_list[-1].shape}", "\n\n")
        # return F.log_softmax(h_list[-1], dim=1)

        node_representation = None
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer):
                node_representation += h_list[layer]

        h_graph = self.pool(node_representation, batch)

        # final predictions
        # print(f"graph: {h_graph}")
        h_graph = self.graph_pred_linear[0](h_graph)
        # print(f"graph: {h_graph}")
        # h_graph = self.graph_norm[0](h_graph)
        h_graph = F.dropout(F.relu(h_graph), self.drop_ratio, training=self.training)

        h_graph = self.graph_pred_linear[1](h_graph)
        # h_graph = self.graph_norm[1](h_graph)
        h_graph = F.dropout(F.relu(h_graph), self.drop_ratio, training=self.training)

        out = self.graph_pred_linear[2](h_graph)
        # print("\n\n", f"forward2: {out.shape}", "\n\n")
        return out

    def __repr__(self):
        return self.__class__.__name__


if __name__ == '__main__':
    model = GNN(10, 5, 3, 64)
    print(model.convs)
