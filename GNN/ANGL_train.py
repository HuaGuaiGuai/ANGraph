# -*- coding:utf-8 -*-

import argparse
import json
import operator
import os
import torch
import numpy as np
import torch.optim as optim
import datetime

from torch_geometric.loader import DataLoader
from functools import reduce
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from evaluate import Evaluator
from ANGL import GNN
from dataset_handcraft_pyg import PygGraphPropPredDataset

reg_criterion = torch.nn.HuberLoss(delta=5)
#reg_criterion = torch.nn.MSELoss()
cls_criterion = torch.nn.BCEWithLogitsLoss()


def tolerance_loss(y_true, y_pred, tolerance=5.0):

    abs_error = torch.abs(y_true - y_pred)


    loss = torch.where(abs_error < tolerance, torch.zeros_like(abs_error), abs_error)

    return torch.mean(loss)

def train(model, device, loader, optimizer, task_type):
    model.train()

    for _, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        batch.x=batch.x.float().to(device)
        pred = model(batch)
        optimizer.zero_grad()
        ## ignore nan targets (unlabeled) when computing training loss.
        is_labeled = batch.y == batch.y
        if "classification" in task_type:
            loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
        else:
            loss = reg_criterion(pred.to(torch.float32), batch.y.to(torch.float32))
            # loss = tolerance_loss(batch.y.to(torch.float32), pred.to(torch.float32), tolerance=5.0)
        loss.backward()
        optimizer.step()


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for _, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict), y_true, y_pred


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GCN, GCN-virtual, GIN, GIN-virtual')
    parser.add_argument('--device', type=int, default=0,
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
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="latency_ap_mnist_4x4_extended",
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
                        help='normilization')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name=args.dataset, root='./data')
    print('This will take some time.......')
    for i in range(len(dataset.data.x)):
      dataset.data.x[i][1:] = dataset.data.x[i][1:]*dataset.data.x[i][0]*2
    dataset.data.y = dataset.data.y / args.norm
    if args.feature == 'full':
        pass
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:, :50]
        dataset.data.edge_attr = dataset.data.edge_attr[:, 0:50]

    split_idx = dataset.get_idx_split([0.8, 0.1, 0.1])

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    model = GNN(num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, net_type=args.gnn, graph_pooling=args.graph_pooling).to(device)
    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    valid_curve = []
    test_curve = []
    train_curve = []

    test_predict_value = []
    test_true_value = []
    valid_predict_value = []
    valid_true_value = []

    save_dir = "{}/{}_{}_{}/{}/{}/pooling_{}/layer{}/{}/".format(args.save_dir, args.dataset, args.emb_dim, args.device,args.norm,args.gnn, args.graph_pooling, args.num_layer,datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
    print(args.graph_pooling)
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
    with open(os.path.join(save_dir, "args.txt"), 'w') as f:
        json.dump(dicts, f, indent=2)

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

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))
    log_f.writelines('Finished training!\n')
    log_f.writelines('Best validation score: {}\n'.format(valid_curve[best_val_epoch]))
    log_f.writelines('Test score: {}\n'.format(test_curve[best_val_epoch]))
    log_f.close()


    f = open(save_dir + args.gnn + '_layer_' + str(args.num_layer) + '.json', 'w')
    result = dict(val=valid_curve[best_val_epoch],
                  test=test_curve[best_val_epoch], train=train_curve[best_val_epoch],
                  test_pred=test_predict_value, valid_pred=valid_predict_value,
                  test_true=test_true_value, valid_true=valid_true_value,
                  train_curve=train_curve, test_curve=test_curve, valid_curve=valid_curve)
    json.dump(result, f)
    f.close()

    if not args.filename == '':
        torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch],
                    'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, f"{save_dir}{args.filename}")

if __name__ == '__main__':
    main()