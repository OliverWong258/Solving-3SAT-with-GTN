# -*- coding: utf-8 -*-
import torch
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from model import network
import warnings
import os
warnings.filterwarnings('ignore')

EPOCHS = 50
EARLY_STOP_CNT = 30


def train(dataset, pos_weight, model_path, embedding_size = 64, n_heads = 1, n_layers = 2, dropout_rate = 0.1, linear_size=128, batch_size = 64):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Training on: ', device)

    # 划分训练集和验证集
    train_set_size = np.ceil(len(dataset) * 0.8)
    valid_set_size = len(dataset) - train_set_size

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [int(train_set_size), int(valid_set_size)])

    # no shuffling, as it is already shuffled
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    model_edge_dim = train_dataset[0].edge_attr.shape[1]

    model = network(feature_size=train_dataset[0].x.shape[1], model_edge_dim=model_edge_dim, embedding_size=embedding_size, 
                n_heads=n_heads, layers=n_layers, dropout=dropout_rate, linear_size=linear_size)
    model = model.to(device)

    weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)

    # 损失函数
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5, amsgrad=False)

    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
         
    final_valid_loss = 50.0    
    final_train_loss = 50.0     
    early_stop_cnt = 0
    loss_difference = 1.0    

    # the following are just for reporting reasons
    valid_loss_list = []
    train_loss_list = []
    stopped = False

    training_loss = 0.0
    validation_loss = 0.0
    
    for epoch in range(EPOCHS):

        print('EPOCH: ', epoch)

        if early_stop_cnt < EARLY_STOP_CNT:
            model.train()
            training_loss = 0.0
            train_step = 0

            for batch in train_loader:
                batch.to(device)
                optimizer.zero_grad()
                
                prediction = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)

                loss = criterion(torch.squeeze(prediction), batch.y.float())

                loss.backward()
                optimizer.step()

                training_loss += loss.item()
                train_step += 1

            training_loss /= train_step
            print("Training Loss: ", training_loss)
            train_loss_list.append(training_loss)

            # 计算验证集上的损失
            model.eval()
            validation_loss = 0.0
            valid_step = 0

            for batch in valid_loader:
                batch.to(device)

                prediction = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)

                loss = criterion(torch.squeeze(prediction), batch.y.float())

                validation_loss += loss.item()
                valid_step += 1
                
            validation_loss /= valid_step
            print("Validation Loss: ", validation_loss)
            valid_loss_list.append(validation_loss)


            # 对比验证集和训练集损失
            if not stopped:
                difference = abs(float(validation_loss) - float(training_loss))
                if difference < loss_difference:
                    loss_difference = difference
                    final_valid_loss = validation_loss
                    final_train_loss = training_loss
                    
                    # 保存当前模型
                    torch.save(model, "./models/"+model_path)

                    early_stop_cnt = 0
                else:
                    early_stop_cnt += 1

            scheduler.step()
        else:
            difference = abs(float(final_valid_loss) - float(final_train_loss))
            print("Early stopping activated, with training and validation loss difference: ", difference, end=" ")
            print("at epoch ", epoch)
            stopped = True
            early_stop_cnt = 0

    print(f"Best training loss: {final_train_loss:.4f}\nBest " f"validation loss: {final_valid_loss:.4f}")

    learning_curve(train_loss_list, valid_loss_list, model_path)
    
    return final_valid_loss


def learning_curve(train_loss, valid_loss, model_path):
    
    epochs = list(range(1, len(train_loss) + 1))

    plt.plot(epochs, train_loss, label='Training Loss', color='r')
    plt.plot(epochs, valid_loss, label='Validation Loss', color='g')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    
    if not os.path.exists("./plots"):
        os.mkdir("./plots")
        
    plt.savefig('./plots/'+model_path+'_train_valid_loss.png')
    plt.close()