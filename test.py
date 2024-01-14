# -*- coding: utf-8 -*-
import torch
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')


def test(testing_dataset, pos_weight, model_path, batch_size=64):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Testing on: {device}')
    
    test_loader = DataLoader(testing_dataset, batch_size=batch_size)
    
    model = torch.load(model_path)
    model.to(device)
    model.eval()
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    test_loss = 0.0
    test_num = 0
    
    predictions = []
    labels = []
    

    for batch in test_loader:
        batch.to(device)
        prediction = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)
        loss = criterion(torch.squeeze(prediction), batch.y.float())
        test_loss += loss.item()
        test_num += 1
        
        predictions.append(np.rint(torch.sigmoid(prediction).cpu().detach().numpy()))
        labels.append(batch.y.cpu().detach().numpy())
        
    predictions = np.concatenate(predictions).ravel()
    labels = np.concatenate(labels).ravel()
    print("labels: ", labels)
    print("predictions: ", predictions)
    print(f"F1 Score  : {f1_score(labels, predictions):.4f}")
    print(f"Accuracy  : {accuracy_score(labels, predictions):.4f}")
    print(f"Precision : {precision_score(labels, predictions):.4f}")
    print(f"Recall    : {recall_score(labels, predictions):.4f}")
    
    return test_loss / test_num

