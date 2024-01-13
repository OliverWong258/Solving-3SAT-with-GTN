# -*- coding: utf-8 -*-
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, \
    RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from dataset import SAT3Dataset
from model import GNN
import warnings
import os
warnings.filterwarnings('ignore')

MAX_NUMBER_OF_EPOCHS = 51
EARLY_STOPPING_COUNTER = 30

embedding_size = 64
n_heads = 1
n_layers = 2
dropout_rate = 0.1
dense_neurons = 128


def plot_errors(errors):
    losses = list(map(list, zip(*errors)))
    train_loss = losses[0]
    valid_loss = losses[1]
    plt.plot([i+1 for i in range(MAX_NUMBER_OF_EPOCHS-1)], train_loss, color='r', label='Training loss')
    plt.plot([i + 1 for i in range(MAX_NUMBER_OF_EPOCHS-1)], valid_loss, color='g', label='Validation loss')

    plt.ylabel('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.title('Early stopping in selected mode')

    plt.legend()
    if not os.path.exists("./plots"):
        os.mkdir("./plots")
    plt.savefig('./plots/train_valid_error.png')
    plt.close()

def training(dataset, pos_weight, model_name, make_err_logs=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Training on: {device}')
    
    # loading the dataset
    dataset = dataset

    # we have already kept a different test set, so split into train and validation 80% - 20%
    train_set_size = np.ceil(len(dataset) * 0.8)
    valid_set_size = len(dataset) - train_set_size

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [int(train_set_size), int(valid_set_size)])

    # no shuffling, as it is already shuffled
    train_loader = DataLoader(train_dataset, batch_size=64)
    valid_loader = DataLoader(valid_dataset, batch_size=64)

    print("Dataset loading completed\n")

    model_edge_dim = train_dataset[0].edge_attr.shape[1]

    # load the GNN model
    print("Model loading...")
    model = GNN(feature_size=train_dataset[0].x.shape[1], model_edge_dim=model_edge_dim, embedding_size=embedding_size, 
                n_heads=n_heads, n_layers=n_layers, dropout_rate=dropout_rate, dense_neurons=dense_neurons)
    model = model.to(device)
    print("Model loading completed\n")

    weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)

    # define a loss function
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5, amsgrad=False)

    # no parameter optimizing for 'scheduler gamma' as it multiplies with weight decay
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # START TRAINING

    # initialize some parameters
    loss_diff = 1.0             # train and validation loss difference, to avoid overfitting
    final_valid_loss = 1000     # validation loss
    final_train_loss = 1000     # training loss
    early_stopping_counter = 0  # counter for early stopping

    # the following are just for reporting reasons
    errors = []
    stopped = False

    for epoch in range(MAX_NUMBER_OF_EPOCHS):

        print(f'EPOCH | {epoch}')

        if early_stopping_counter < EARLY_STOPPING_COUNTER:
            # perform one training epoch
            model.train()
            training_loss = 0.0
            step = 0
            for batch in train_loader:
                batch.to(device)
                optimizer.zero_grad()
                # make prediction
                prediction = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)
                # calculate loss
                loss = criterion(torch.squeeze(prediction), batch.y.float())
                # calculate gradient
                loss.backward()
                optimizer.step()

                training_loss += loss.item()
                step += 1

            training_loss /= step
            print(f"Training Loss   : {training_loss:.4f}")

            # compute validation set loss
            model.eval()
            validation_loss = 0.0
            step = 0
            for batch in valid_loader:
                batch.to(device)
                # make prediction
                prediction = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)
                # calculate loss
                loss = criterion(torch.squeeze(prediction), batch.y.float())

                validation_loss += loss.item()
                step += 1
            validation_loss /= step
            print(f"Validation Loss : {validation_loss:.4f}\n")

            errors += [(training_loss, validation_loss)]

            # check for early stopping if model is yet to finish its training
            if not stopped:
                difference = abs(float(validation_loss) - float(training_loss))
                if difference < loss_diff:
                    loss_diff = difference
                    final_valid_loss = validation_loss
                    final_train_loss = training_loss
                    # if still some progress can be made -> save the currently best model
                    torch.save(model, model_name)

                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

            scheduler.step()
        else:
            difference = abs(float(final_valid_loss) - float(final_train_loss))
            print(f"Early stopping activated, with training and validation loss difference: {difference:.4f}")
            stopped = True
            early_stopping_counter = 0

    print(f"Finishing training with best training loss: {final_train_loss:.4f} and best "
          f"validation loss: {final_valid_loss:.4f}")

    if make_err_logs:
        plot_errors(errors)

    return final_valid_loss
