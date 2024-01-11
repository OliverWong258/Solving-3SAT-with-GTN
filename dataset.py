from torch_geometric.data import Dataset, Data
from tqdm import tqdm
import numpy as np
import torch
import os
from dataprocessing import processed_data

class dataset(Dataset):
    def __init__(self, root, df, test, transform=None, pre_transform=None):
        self.df = df
        self.test = test
        self.data = None
        super(dataset, self).__init__(root, transform, pre_transform)
        

    def process_data(self):
        print("processing data for dataset...")
        self.df = self.df.reset_index()
        
        for index, cnf in tqdm(self.df.iterrows(), total = self.df.shape[0]):
            # node features
            node_features = torch.tensor(cnf["nodeFeatures"], dtype=torch.float)
            # edges
            edge_index = torch.tensor(cnf["edges"], dtype=torch.long)
            edge_num = edge_index.size(dim=1)
            # edge features
            edge_features = torch.tensor(cnf["edgeAttr"], dtype=torch.float).view(edge_num, -1)
            """
            print("edge_index: ", edge_index)
            print("edge_features: ", edge_features)
            """
            # labels
            label = torch.tensor(np.asarray(cnf["label"]), dtype=torch.int64)
            # create data
            data = Data(x=node_features, edge_index=edge_index,edge_attr=edge_features, y=label)
            
            if not os.path.exists(self.processed_dir):
                os.makedirs(self.processed_dir)
                print(f"Created directory: {self.processed_dir}")
                
            if self.test:
                torch.save(data, os.path.join(self.processed_dir, f'test_data_{index}.pt'))
            else:
                torch.save(data, os.path.join(self.processed_dir, f'train_data_{index}.pt'))

    def len(self):
        return self.data.shape[0]
    
    def get(self, index):
        if self.test:
            return torch.load(os.path.join(self.processed_dir, f'test_data_{index}.pt'))
        else:
            return torch.load(os.path.join(self.processed_dir, f'train_data_{index}.pt'))

"""
test_data = processed_data("./datatest")
test_data.process_rawdata()
test_dataset = dataset(root="./", df=test_data.df_train, test = False)
test_dataset.process_data()
print(test_dataset.get(0))
"""