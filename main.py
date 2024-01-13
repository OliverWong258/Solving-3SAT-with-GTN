# -*- coding: utf-8 -*-
import sys
from train import train
from test import test
import os
from dataset import SAT3Dataset

def main():
    if not os.path.exists("./processed"):
        os.mkdir("./processed")
        
    if not os.path.exists("./models"):
        os.mkdir("./models")
        
    if len(sys.argv) < 3:
        print("Usage: python main.py <operation>")
        sys.exit(1)
        
    operation = sys.argv[1]
    model_path = sys.argv[2]
    model_path = "./models/" + model_path
    
    if operation == "train":
        dataset = SAT3Dataset(root="./")
        train(dataset=dataset, pos_weight=dataset.pos_weight, model_path=model_path)
        
    elif operation == "test":
        dataset = SAT3Dataset(root="./", test=True)
        test(testing_dataset=dataset, pos_weight=dataset.pos_weight, model_path=model_path)



if __name__ == "__main__":
    main()




