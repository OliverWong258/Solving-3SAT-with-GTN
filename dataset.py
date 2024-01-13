# -*- coding: utf-8 -*-
from torch_geometric.data import Dataset, Data
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import os
from dataprocessing import process_raw


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
        return self.filename

    def process(self):
        raw_data = process_raw(directory=os.path.join(self.root, self.filename))
        self.pos_weight = raw_data.dataset_processing()
        if not self.test:
            self.data = raw_data.df_tr
        else:
            self.data = raw_data.df_test
        
        print("Dataset loading...")
        for index, cnf in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            # get node features (here we actually don't have many)
            node_feats = torch.tensor(cnf["variablesSymb"], dtype=torch.float)
            # get adjacency info
            edge_index = torch.tensor(cnf["edges"], dtype=torch.long)
            num_edges = edge_index.size(dim=1)
            # get edge features |  view is used in order to get the correct dimensions as specified by COO format
            edge_feats = torch.tensor(cnf["edgeAttr"], dtype=torch.float).view(num_edges, -1)
            # get labels info
            label = torch.tensor(np.asarray(cnf["label"]), dtype=torch.int64)
            # now, create data object
            data = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_feats, y=label)
            # save the data
            if self.test:
                torch.save(data, os.path.join(self.processed_dir, f'data_test_{index}.pt'))
            else:
                torch.save(data, os.path.join(self.processed_dir, f'data_{index}.pt'))
                

    def len(self):
        return self.data.shape[0]

    def get(self, index):
        if self.test:
            return torch.load(os.path.join(self.processed_dir, f'data_test_{index}.pt'))
        else:
            return torch.load(os.path.join(self.processed_dir, f'data_{index}.pt'))