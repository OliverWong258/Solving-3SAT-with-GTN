# -*- coding: utf-8 -*-
import sys

from torch import embedding
from train import train
from test import test
import os
from dataset import SAT3Dataset
import argparse
from data_process import process_raw

def main():
    if not os.path.exists("./processed"):
        os.mkdir("./processed")
        
    if not os.path.exists("./models"):
        os.mkdir("./models")
        
    if len(sys.argv) < 2:
        print("Usage: python main.py <model_path>")
        sys.exit(1)
        
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', help="model path", required=True)
    parser.add_argument('--d', help='data path', required=False)
    parser.add_argument('--e', help="embedding size", required=False)
    parser.add_argument('--h', help="heads num", required=False)
    parser.add_argument('--l', help="layers num", required=False)
    parser.add_argument('--r', help="dropout rate", required=False)
    parser.add_argument('--ls', help="linear size", required=False)
    parser.add_argument('--b', help="batch size", required=False)
    
    args = parser.parse_args()
    model_path = args.m
    data_path = args.d if args.d != None else './data'
    embedding_size = args.e if args.e != None else 64
    n_heads = args.h if args.h != None else 1
    n_layers = args.l if args.l != None else 2
    dropout_rate = args.r if args.r != None else 0.1
    linear_size = args.ls if args.ls != None else 128
    batch_size = args.b if args.b != None else 64
    

    raw_data = process_raw(directory=data_path)
    pos_weight = raw_data.dataset_processing()
    
    
    # 训练模型
    dataset = SAT3Dataset(root="./", raw_data=raw_data, test=False)
    train(dataset=dataset, pos_weight=pos_weight, model_path=model_path, embedding_size=embedding_size, n_heads=n_heads, n_layers=n_layers,
            dropout_rate=dropout_rate, linear_size=linear_size, batch_size=batch_size)
        
    # 测试模型
    dataset = SAT3Dataset(root="./", raw_data=raw_data, test=True)
    test(testing_dataset=dataset, pos_weight=pos_weight, model_path=model_path, batch_size=batch_size)



if __name__ == "__main__":
    main()




