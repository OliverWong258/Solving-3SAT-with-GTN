# -*- coding: utf-8 -*-
from dataprocessing import process_raw
import sys
from train import training
from test import test
import os, shutil
from dataset import SAT3Dataset


def delete_folder_contents(folders):
    for folder in folders:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

def main():
    delete_folder_contents(["./processed"])
    raw_data = process_raw()
    pos_weight = raw_data.dataset_processing()
    dataset = SAT3Dataset(root="./", dataframe=raw_data.df)
    training(dataset=dataset, pos_weight=pos_weight, model_name='./final_model_same_sets.pth', make_err_logs=True)



if __name__ == "__main__":
    main()




