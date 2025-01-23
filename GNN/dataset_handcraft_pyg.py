from torch_geometric.data import InMemoryDataset
import pandas as pd
import os
import os.path as osp
import torch
import numpy as np
import random

from read_graph import read_graph_pyg


class PygGraphPropPredDataset(InMemoryDataset):
    def __init__(self, name, root='dataset', transform=None, pre_transform=None, meta_dict=None):
        """
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects

            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        """

        self.name = name  # original name, e.g., ogbg-molhiv

        if meta_dict is None:
            self.dir_name = '_'.join(name.split('-'))

            # check if previously-downloaded folder exists.
            # If so, use that one.
            if osp.exists(osp.join(root, self.dir_name + '_pyg')):
                self.dir_name = self.dir_name + '_pyg'

            self.original_root = root
            self.root = osp.join(root, self.dir_name)

            # master = pd.read_csv(os.path.join(os.path.dirname(__file__), 'datasets.csv'), index_col=0)
            master = pd.read_csv('./datasets.csv', index_col=0)
            if not self.name in master:
                error_mssg = 'Invalid dataset name {}.\n'.format(self.name)
                error_mssg += 'Available datasets are as follows:\n'
                error_mssg += '\n'.join(master.keys())
                raise ValueError(error_mssg)
            self.meta_info = master[self.name]

        else:
            self.dir_name = meta_dict['dir_path']
            self.original_root = ''
            self.root = meta_dict['dir_path']
            self.meta_info = meta_dict

        # self.download_name = self.meta_info['download_name']  # name of downloaded file, e.g., tox21

        # self.num_tasks = int(self.meta_info['num tasks'])
        self.eval_metric = self.meta_info['evaluation_metric']
        self.task_type = self.meta_info['task_type']
        self.__num_classes__ = int(self.meta_info['num_classes'])
        self.binary = False
        self.rand_split_seed = int(self.meta_info['rand_split_seed'])
        self.num_samples = int(self.meta_info['num_samples'])

        super(PygGraphPropPredDataset, self).__init__(self.root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.data.x = self.data.x.float()

    def get_idx_split(self, ratios):

        path = osp.join(self.root, 'split', 'random')
        if not os.path.exists(path):
            os.makedirs(path)

        if not os.path.isfile(os.path.join(path, 'train.csv.gz')):
            train_idx, valid_idx, test_idx = self._get_plit_idx(self.num_samples, ratios, self.rand_split_seed)
            pd.DataFrame(train_idx).to_csv(osp.join(path, 'train.csv.gz'), compression='gzip', index=False,
                                           header=False)
            pd.DataFrame(valid_idx).to_csv(osp.join(path, 'valid.csv.gz'), compression='gzip', index=False,
                                           header=False)
            pd.DataFrame(test_idx).to_csv(osp.join(path, 'test.csv.gz'), compression='gzip', index=False, header=False)

        train_idx = pd.read_csv(osp.join(path, 'train.csv.gz'), compression='gzip', header=None).values.T[0]
        valid_idx = pd.read_csv(osp.join(path, 'valid.csv.gz'), compression='gzip', header=None).values.T[0]
        test_idx = pd.read_csv(osp.join(path, 'test.csv.gz'), compression='gzip', header=None).values.T[0]

        return {'train': torch.tensor(train_idx, dtype=torch.long), 'valid': torch.tensor(valid_idx, dtype=torch.long),
                'test': torch.tensor(test_idx, dtype=torch.long)}

    def _get_plit_idx(self, num_samples, ratios, rand_split_seed):
        sample_list = list(range(num_samples))
        if rand_split_seed is not None:
            random.seed(rand_split_seed)

        assert sum(ratios) == 1, f"Wrong ratios: {ratios}"

        len_train = int(num_samples * ratios[0])
        len_val = int(num_samples * ratios[1])

        random.shuffle(sample_list)

        train_idx = sample_list[:len_train]
        valid_idx = sample_list[len_train:len_train + len_val]
        test_idx = sample_list[len_train + len_val:]

        print(num_samples, len(train_idx), len(valid_idx), len(test_idx))
        return train_idx, valid_idx, test_idx

    @property
    def num_classes(self):
        return self.__num_classes__

    @property
    def raw_file_names(self):
        if self.binary:
            return ['data.npz']
        else:
            file_names = ['edge']
            if self.meta_info['has_node_attr'] == 'True':
                file_names.append('node-feat')
            if self.meta_info['has_edge_attr'] == 'True':
                file_names.append('edge-feat')
            return [file_name + '.csv.gz' for file_name in file_names]

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def process(self):
        # read pyg graph list
        add_inverse_edge = self.meta_info['add_inverse_edge'] == 'True'

        data_list = read_graph_pyg(self.raw_dir, add_inverse_edge=add_inverse_edge)

        if self.task_type == 'subtoken prediction':
            graph_label_notparsed = pd.read_csv(osp.join(self.raw_dir, 'graph-label.csv.gz'), compression='gzip',
                                                header=None).values
            graph_label = [str(graph_label_notparsed[i][0]).split(' ') for i in range(len(graph_label_notparsed))]

            for i, g in enumerate(data_list):
                g.y = graph_label[i]

        else:
            if self.binary:
                graph_label = np.load(osp.join(self.raw_dir, 'graph-label.npz'))['graph_label']
            else:
                graph_label = pd.read_csv(osp.join(self.raw_dir, 'graph-label.csv.gz'), compression='gzip',
                                          header=None).values

            has_nan = np.isnan(graph_label).any()

            for i, g in enumerate(data_list):
                if 'classification' in self.task_type:
                    if has_nan:
                        g.y = torch.from_numpy(graph_label[i]).view(1, -1).to(torch.float32)
                    else:
                        g.y = torch.from_numpy(graph_label[i]).view(1, -1).to(torch.long)
                else:
                    print(type(g),g)
                    g.y = torch.from_numpy(graph_label[i]).view(1, -1).to(torch.float32)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        # self.data = data
        self.slices = slices

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    # pyg_dataset = PygGraphPropPredDataset(name = 'ogbg-molpcba')
    # print(pyg_dataset.num_classes)
    # split_index = pyg_dataset.get_idx_split()
    # print(pyg_dataset)
    # print(pyg_dataset[0])
    # print(pyg_dataset[0].y)
    # print(pyg_dataset[0].y.dtype)
    # print(pyg_dataset[0].edge_index)
    # print(pyg_dataset[split_index['train']])
    # print(pyg_dataset[split_index['valid']])
    # print(pyg_dataset[split_index['test']])

    pyg_dataset = PygGraphPropPredDataset(root='./data/', name='latency_sparse_noc')
    print(f"pyg_dataset.num_classes: {pyg_dataset.num_classes}")
    print(f"pyg_dataset[1]: {pyg_dataset[1]}")
    split_index = pyg_dataset.get_idx_split([0.8, 0.1, 0.1])
    # print(pyg_dataset[0].node_is_attributed)
    # print([pyg_dataset[i].x[1] for i in range(100)])
    # print(pyg_dataset[0].y)
    print(f"pyg_dataset[1].x: {pyg_dataset[1].x}")
    print(f"pyg_dataset[split_index['train']]: {pyg_dataset[split_index['train']]}")
    print(f"pyg_dataset[split_index['valid']]: {pyg_dataset[split_index['valid']]}")
    print(f"pyg_dataset[split_index['test']]: {pyg_dataset[split_index['test']]}")

    print(f"pyg_dataset.slices: {pyg_dataset.slices}")

    from torch_geometric.loader import DataLoader

    loader = DataLoader(pyg_dataset, batch_size=32, shuffle=False)
    for batch in loader:
        print(batch.batch)
        print(batch)
        print(len(batch.y))

        break
