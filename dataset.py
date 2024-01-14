# -*- coding: utf-8 -*-
from torch_geometric.data import Dataset, Data
from tqdm import tqdm
import numpy as np
import torch
import os, shutil
from data_process import process_raw


class SAT3Dataset(Dataset):
    def __init__(self, root, raw_data, test=False, transform=None, pre_transform=None):
        self.root = root
        self.filename = "File is not required"
        self.raw_data = raw_data
        self.test = test
        self.data = None
        super(SAT3Dataset, self).__init__(root, transform, pre_transform)
    
    @property
    def processed_file_names(self):
        # Dataset类要求填充成员函数，并无实际意义
        return self.filename

    def process(self):
        if not self.test:
            self.data = self.raw_data.df_train.reset_index()
        else:
            self.data = self.raw_data.df_test.reset_index()
        
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
        
    def delete_folder_contents(self):
        for filename in os.listdir(self.processed_dir):
            file_path = os.path.join(self.processed_dir, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                
        print("Delete files in ", self.processed_dir)