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
        
    if len(sys.argv) < 3:
        print("Usage: python main.py --m <model_path> --s <separate>")
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
    parser.add_argument('--s', help="separate", required=True)
    
    args = parser.parse_args()
    model_path = args.m
    data_path = args.d if args.d != None else './data'
    embedding_size = int(args.e) if args.e != None else int(64)
    n_heads = int(args.h) if args.h != None else int(1)
    n_layers = int(args.l) if args.l != None else int(2)
    dropout_rate = float(args.r) if args.r != None else float(0.1)
    linear_size = int(args.ls) if args.ls != None else int(128)
    batch_size = int(args.b) if args.b != None else int(64)
    separate_test = False if args.s == 0 else True
    print(separate_test)
    

    raw_data = process_raw(directory=data_path, separate=separate_test)
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




