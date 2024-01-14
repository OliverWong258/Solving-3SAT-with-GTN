# -*- coding: utf-8 -*-
import torch
from torch.nn import Linear, BatchNorm1d, ModuleList, ELU
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class network(torch.nn.Module):
    def __init__(self, feature_size, model_edge_dim, embedding_size, n_heads, layers, dropout, linear_size):
        super(network, self).__init__()
        self.feature_size = feature_size        # 初始节点特征维度
        self.edge_dim = model_edge_dim          # 边特征维度
        self.n_layers = layers                  # 循环层数
        self.embedding_size = embedding_size    # 图卷积层映射维度
        self.n_heads = n_heads                  # 注意力头数
        self.dropout = dropout                  
        self.linear_size = linear_size          # 末尾全连接层维度
        self.elu = ELU()
        
        
        self.init_conv_layer = TransformerConv(self.feature_size, self.embedding_size, heads=self.n_heads, 
                                               dropout=self.dropout, edge_dim=self.edge_dim, beta=True)
        self.init_linear_layer = Linear(self.embedding_size * self.n_heads, self.embedding_size)
        self.init_bn_layer = BatchNorm1d(self.embedding_size)

        self.conv_layers = ModuleList([])       # 图卷积层
        self.linear_layers = ModuleList([])     # 全连接层
        self.bn_layers = ModuleList([])         # 归一化层
        
        for _ in range(self.n_layers):
            self.conv_layers.append(TransformerConv(self.embedding_size, self.embedding_size, heads=self.n_heads, 
                                                    dropout=self.dropout, edge_dim=self.edge_dim, beta=True))
            self.linear_layers.append(Linear(self.embedding_size * self.n_heads, self.embedding_size))
            self.bn_layers.append(BatchNorm1d(self.embedding_size))

        # final linear layers
        self.linear_layer1 = Linear(embedding_size * 2, linear_size)
        self.linear_layer2 = Linear(linear_size, int(linear_size / 2))
        self.linear_layer3 = Linear(int(linear_size / 2), 1)

    def forward(self, x, edge_attr, edge_index, batch_index):
        x = self.init_conv_layer(x, edge_index, edge_attr)
        x = self.elu(self.init_linear_layer(x))
        x = self.init_bn_layer(x)

        # holds the intermediate graph representations
        global_representation = []
        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = self.elu(self.linear_layers[i](x))
            x = self.bn_layers[i](x)

            global_representation.append(torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1))

        x = sum(global_representation)

        # output block
        x = self.elu(self.linear_layer1(x))
        x = self.elu(self.linear_layer2(x))
        x = self.linear_layer3(x)

        return x
