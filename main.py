# -*- coding: utf-8 -*-
from dataprocessing import dataset_processing
import sys
from dataset import dataset
from train import training
from test import test
import os, shutil


def delete_folder_contents(folders):
    for folder in folders:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

def main():
    delete_folder_contents(["./raw", "./processed"])
    pos_weight = dataset_processing(separate_test=False)
    training(model_name='./final_model_same_sets.pth', make_err_logs=True)



if __name__ == "__main__":
    main()




