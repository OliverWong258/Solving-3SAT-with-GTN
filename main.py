# -*- coding: utf-8 -*-
from dataprocessing import processed_data
import sys
from dataset import dataset
from train import train
from test import test
import os

data_path = "./data"
seperate = False

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <operation>")
        sys.exit(1)    
    operation = sys.argv[1]
    model_path = "./models/" + sys.argv[2]
    if not os.path.exists("./models"):
        os.makedirs("./models")
        print("\nCreate ./models")
    
    data = processed_data(data_path=data_path, seperate=seperate)
    pos_weight = data.process_rawdata()
    
    
    if operation == "train":
        train_dataset = dataset(root="./", df=data.df_train, test = False)
        train_dataset.process_data()
        
        train_loss, valid_loss = train(dataset=train_dataset, pos_weight=pos_weight, model_path=model_path)
        print("Final training loss: ", train_loss)
        print("Final validation loss: ", valid_loss)
        
    elif operation == "test":
        test_dataset = dataset(root="./", df=data.df_test, test = True)
        test_dataset.process_data()
        
        test_loss = test(model_path=model_path, testing_dataset=test_dataset, pos_weight=pos_weight)
        print("Test loss: ", test_loss)



if __name__ == "__main__":
    main()




