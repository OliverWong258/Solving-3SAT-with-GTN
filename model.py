# -*- coding: utf-8 -*-
import torch
from torch.nn import Linear, BatchNorm1d, ModuleList, Embedding
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class network(torch.nn.Module):
    def __init__(self, value_dim, feature_size, embedding_size, attention_heads, layers, dropout_rate, linear_size, edge_dim):
        super(network, self).__init__()
        self.value_dim = value_dim                  # 节点初始特征的维度
        self.feature_size = feature_size            # 嵌入后的节点特征维度
        self.embedding_size = embedding_size        # 图卷积层中变换后的维度
        self.attention_heads = attention_heads      # 注意力头数
        self.layers = layers                        # 模型堆叠层数
        self.dropout_rate = dropout_rate            
        self.linear_size = linear_size              # 最后全连接层的维度
        self.edge_dim = edge_dim                    # 边特征维度
        
        self.conv_layers = ModuleList([])
        self.linear_layers = ModuleList([])
        self.bn_layers = ModuleList([])
        
        # 嵌入层，将节点特征映射到更高维空间
        self.embedding_layer = Embedding(self.value_dim, self.feature_size, dtype=torch.float32)
        self.conv_layer = TransformerConv(self.feature_size, self.embedding_size, heads = self.attention_heads, 
                                          dropout=self.dropout_rate, edge_dim=self.edge_dim, beta=True)
        self.linear_layer = Linear(self.embedding_size*self.attention_heads, self.embedding_size)
        self.bn_layer = BatchNorm1d(self.embedding_size)
        
        for _ in range(self.layers):
            self.conv_layers.append(TransformerConv(self.embedding_size, self.embedding_size, heads=self.attention_heads,
                                                    dropout=self.dropout_rate, edge_dim=self.edge_dim, beta=True))
            self.linear_layers.append(Linear(self.embedding_size*self.attention_heads, self.embedding_size))
            self.bn_layers.append(BatchNorm1d(self.embedding_size))
            
        self.linear1 = Linear(self.embedding_size*2, self.linear_size)
        self.linear2 = Linear(self.linear_size, int(self.linear_size/2))
        self.linear3 = Linear(int(self.linear_size/2), 1)
        
    def forward(self, x, edge_attr, edge_index, batch_index):
        x = x.to(torch.int)
        node_num = x.shape[0]
        x = self.embedding_layer(x)
        #print("x after embedding: ", x)
        x = x.view(node_num,-1)
        #print("x after view: ", x)
        x = self.conv_layer(x, edge_index, edge_attr)
        x = torch.relu(self.linear_layer(x))
        x = self.bn_layer(x)
        #print("x after first unit: ", x)
        global_representation = []
        for i in range(self.layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = torch.relu(self.linear_layers[i](x))
            x = self.bn_layers[i](x)
            
            # 最大池化和平均池化
            global_representation.append(torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1))
        #print("x after second unit: ", x)
        #print("global_representation: ", global_representation)    
        x = sum(global_representation)
        #print("x after sum: ", x)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        
        return x
    

