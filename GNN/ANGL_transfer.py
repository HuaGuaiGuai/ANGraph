import argparse
import json
import operator
import os, sys
from functools import reduce
import numpy as np
import datetime

import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from evaluate import Evaluator
from ANGL import GNN
from dataset_handcraft_pyg import PygGraphPropPredDataset
from ANGL_train import train
from ANGL_train import eval

reg_criterion = torch.nn.HuberLoss(delta=5)
cls_criterion = torch.nn.BCEWithLogitsLoss()

parser = argparse.ArgumentParser(description='GCN, GCN-virtual, GIN, GIN-virtual')
parser.add_argument('--device', type=int, default=3,
                        help='which gpu to use if any (default: 0)')
parser.add_argument('--gnn', type=str, default='sage',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual), sage, pna, rgcn, gat')
parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.3)')
parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
parser.add_argument('--emb_dim', type=int, default=100,
                        help='dimensionality of hidden units in GNNs (default: 300)')
parser.add_argument('--batch_size', type=int, default=1,
                        help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train (default: 300)')
parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
parser.add_argument('--dataset', type=str, default="latncey_mnist_mix_nonuniform",
                        help='dataset name')
parser.add_argument('--graph_pooling', type=str, default="sum",
                        help='sum, mean, max, attention')
parser.add_argument('--save_dir', type=str, default="logs",
                        help='dir of saved models')
parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
parser.add_argument('--filename', type=str, default="train.tensor",
                        help='filename to output result (default: )')
parser.add_argument('--train_log', type=str, default="train",
                        help='log file')
parser.add_argument('--norm', type=str, default=1000,
                        help='normalization factor')
args = parser.parse_args(args=[])

device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### automatic dataloading and splitting
'''
dataset:
latency_ap_mnist_tsmc28
latency_ap_mnist_tsmc65
latency_ap_mnist_tsmc22
'''
transfer_dataset = "latency_ap_mnist_tsmc28"

dataset = PygGraphPropPredDataset(transfer_dataset, root='./data')
evaluator = Evaluator(transfer_dataset)
print('This will take some time.......')
for i in range(len(dataset.data.x)):
    dataset.data.x[i][1:] = dataset.data.x[i][1:]*dataset.data.x[i][0]*2
dataset.data.y = dataset.data.y / 1000

split_idx = dataset.get_idx_split([0.1, 0.1, 0.8])

train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)


model = GNN(num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, net_type=args.gnn, graph_pooling=args.graph_pooling).to(device)
layers_to_freeze = [model.node_emb,  model.edge_emb, model.convs, model.batch_norms, model.graph_norm]

for layer in layers_to_freeze:
    for param in layer.parameters():
        param.requires_grad = False

for name, param in model.named_parameters():
    print(f"{name}: requires_grad = {param.requires_grad}")

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
model.load_state_dict(torch.load("./saved_model/latency_scale.pt")['model_state_dict'])


valid_curve = []
test_curve = []
train_curve = []

test_predict_value = []
test_true_value = []
valid_predict_value = []
valid_true_value = []

save_dir = "{}/{}_{}_{}/{}/{}/pooling_{}/layer{}/{}/".format(args.save_dir, args.dataset, args.emb_dim, args.device,args.norm,args.gnn, args.graph_pooling, args.num_layer,datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
os.makedirs(save_dir, exist_ok=True)
train_log = save_dir + args.train_log + '.log'
log_f = open(train_log, 'w')
dicts = {
    "gnn_type": "{}".format(args.gnn),
    "drop_ratio": "{}".format(args.drop_ratio),
    "num_layer": "{}".format(args.num_layer),
    "emb_dim": "{}".format(args.emb_dim),
    "batch_size": "{}".format(args.batch_size),
    "epochs": "{}".format(args.epochs),
    "feature": "{}".format(args.feature),
    "dataset": "{}".format(args.dataset),
    "graph_pooling": "{}".format(args.graph_pooling)
}
for epoch in range(1, args.epochs + 1):
    print("=====Epoch {}".format(epoch))
    print('Training...')
    train(model, device, train_loader, optimizer, dataset.task_type)

    print('Evaluating...')
    # train_perf, _, _ = eval(model, device, train_loader, evaluator)
    valid_perf, v_true, v_pred = eval(model, device, valid_loader, evaluator)
    train_perf = valid_perf
    test_perf, t_true, t_pred = eval(model, device, test_loader, evaluator)

    print({'cuda': args.device,'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})
    log_f.writelines(f"Epoch {epoch} -> Train: {train_perf}, Validation: {valid_perf}, Test: {test_perf}\n")

    train_curve.append(train_perf[dataset.eval_metric])
    valid_curve.append(valid_perf[dataset.eval_metric])
    test_curve.append(test_perf[dataset.eval_metric])

    test_predict_value.append(reduce(operator.add, t_pred.tolist()))
    valid_predict_value.append(reduce(operator.add, v_pred.tolist()))

    test_loss = test_perf[dataset.eval_metric]
    if test_loss >= np.max(np.array(test_curve)):
        PATH = save_dir + args.gnn + '_layer_' + str(args.num_layer) + args.dataset  + '_model.pt'
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': test_loss
                    }, PATH)

test_true_value = reduce(operator.add, t_true.tolist())
valid_true_value = reduce(operator.add, v_true.tolist())
