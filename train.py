# -*- coding: utf-8 -*-
import torch
from torch_geometric.loader import DataLoader
import numpy as np
from dataset import dataset
from model import network
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 超参数
batch_size = 64
embedding_size = 64
attention_heads = 1
layers = 2
dropout_rate = 0.1
linear_size = 128
learning_rate = 0.01
weight_decay = 1e-5

EPOCHS = 51
EARLY_STOP = 15

def train(dataset, pos_weight, model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'\nTraining on: {device}')
    
    # 划分训练集和验证集
    train_size = np.ceil(len(dataset) * 0.8)
    valid_size = len(dataset) - train_size
    print("\nTrain size: ", train_size)
    print("Valid size: ", valid_size)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [int(train_size), int(valid_size)])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    
    feature_size = train_dataset[0].x.shape[1]
    edge_dim = train_dataset[0].edge_attr.shape[1]
    print("feature_size: ", feature_size)
    print("edge_dim: ", edge_dim)
    model = network(feature_size=feature_size, embedding_size=embedding_size, attention_heads=attention_heads, 
                    layers=layers, dropout_rate=dropout_rate, linear_size=linear_size, edge_dim=edge_dim)
    model = model.to(device)
    
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=False)
    # 设置学习率衰减
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    

    loss_difference = 1.0             # 训练集和验证集的误差差距，避免过拟合
    final_valid_loss = 1000.0  
    final_train_loss = 1000.0 
    current_train_loss = 0.0
    current_valid_loss = 0.0
    early_stop_cnt = 0                # 早停计数器
    
    train_loss_list = []
    valid_loss_list = []
    stopped = False
    for epoch in range(EPOCHS):
        print("Epoch: ", epoch)
        
        if(early_stop_cnt < EARLY_STOP):
            model.train()
            
            current_train_loss = 0.0
            train_batch = 0
            for batch in train_loader:
                batch.to(device)
                optimizer.zero_grad()
                prediction = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)
                loss = criterion(torch.squeeze(prediction), batch.y.float())
                loss.backward()
                optimizer.step()
        
                current_train_loss += loss.item()
                train_batch += 1
                
            current_train_loss /= train_batch
            print("Training loss: ", current_train_loss)
            train_loss_list.append(current_train_loss)
            
            # 在验证集上测试
            model.eval()
            current_valid_loss = 0.0
            valid_batch = 0
            for batch in valid_loader:
                batch.to(device)
                prediction = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)
                loss = criterion(torch.squeeze(prediction), batch.y.float())
                
                current_valid_loss += loss.item()
                valid_batch += 1
                
            current_valid_loss /= valid_batch
            print("Valid loss: ", current_valid_loss)
            valid_loss_list.append(current_valid_loss)
            
            if not stopped:
                difference = abs(float(current_train_loss) - float(current_valid_loss))
                if difference < loss_difference:
                    loss_difference = difference
                    final_valid_loss = current_valid_loss
                    final_train_loss = current_train_loss
                    # 保存当前模型
                    torch.save(model, model_path)
                    early_stop_cnt = 0
                else:
                    early_stop_cnt += 1
                    
            scheduler.step()
        else:
            # 触发早停机制
            difference = abs(float(current_train_loss) - float(current_valid_loss))
            stopped = True
            early_stop_cnt = 0
            print("Early stop activated with loss difference: ", difference, end=' ')
            print("at epoch ", epoch)
     
    learning_curve(train_loss_list, valid_loss_list)
    return final_train_loss, final_valid_loss
                    

def learning_curve(train_loss, valid_loss):
    epochs = list(range(1, len(train_loss) + 1))

    # 绘制训练集和验证集的学习曲线
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, valid_loss, label='Validation Loss')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss per epoch')

    plt.legend()
    plt.show()