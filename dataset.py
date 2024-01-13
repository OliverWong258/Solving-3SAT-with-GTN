# -*- coding: utf-8 -*-
from torch_geometric.data import Dataset, Data
from tqdm import tqdm
import numpy as np
import torch
import os
from data_process import process_raw


class SAT3Dataset(Dataset):
    def __init__(self, root, filename="data", test=False, transform=None, pre_transform=None):
        self.root = root
        self.filename = filename
        self.test = test
        self.data = None
        self.pos_weight = 0
        super(SAT3Dataset, self).__init__(root, transform, pre_transform)
    
    @property
    def processed_file_names(self):
        # Dataset类要求填充成员函数，并无实际意义
        return self.filename

    def process(self):
        raw_data = process_raw(directory=os.path.join(self.root, self.filename))
        self.pos_weight = raw_data.dataset_processing()
        
        if not self.test:
            self.data = raw_data.df_train.reset_index()
        else:
            self.data = raw_data.df_test.reset_index()
        
        print("Loading dataset")
        for index, cnf in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            # 节点信息
            node_feats = torch.tensor(cnf["node_features"], dtype=torch.float)
            # 边信息
            edge_index = torch.tensor(cnf["edges"], dtype=torch.long)
            num_edges = edge_index.size(dim=1)
            edge_feats = torch.tensor(cnf["edge_attr"], dtype=torch.float).view(num_edges, -1)
            # 标签信息
            label = torch.tensor(np.asarray(cnf["label"]), dtype=torch.int64)
            data = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_feats, y=label)
            if self.test:
                torch.save(data, os.path.join(self.processed_dir, f'data_test_{index}.pt'))
            else:
                torch.save(data, os.path.join(self.processed_dir, f'data_train_{index}.pt'))
                

    def len(self):
        return self.data.shape[0]

    def get(self, index):
        if self.test:
            return torch.load(os.path.join(self.processed_dir, f'data_test_{index}.pt'))
        else:
            return torch.load(os.path.join(self.processed_dir, f'data_train_{index}.pt'))