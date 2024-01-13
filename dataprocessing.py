# -*- coding: utf-8 -*-
import os
import pandas as pd
from tqdm import tqdm
import random
import numpy as np


dictionary = {"nodeFeatures": [],
                "edges": [],
                "edgeAttr": [],
                "label": []}

# store the processed data
class processed_data():
    def __init__(self, data_path, seperate = False, frac = 0.8):
        self.data_path = data_path  # path of the raw data
        self.seperate = seperate    # whether the train and test data are seperated or not
        self.sat = 0            # the number of satisfiable 
        self.unsat = 0          # the number of unsatisfiable
        self.df = pd.DataFrame(dictionary)
        self.df_train = pd.DataFrame(dictionary)
        self.df_test = pd.DataFrame(dictionary)
        self.df = self.df.astype('object')
        self.df_train = self.df_train.astype('object')
        self.df_test = self.df_test.astype('object')
        self.frac = frac
        
    def process_rawdata(self):
        print("Processing raw data")
        for dir in tqdm(os.listdir(self.data_path)):
            current_dir = self.data_path + "/" + dir
            if os.path.isdir(current_dir):
                # read messeage from the name of the directory
                info = dir.split(".")
                var_num = int(info[0][2:]) if (info[0][1] == "F") else int(info[0][3:]) # number of variables
                clause_num = int(info[1])
                label = 1 if(info[0][1] == "F") else 0  # satisfiable or not
                
                if not (self.seperate and (info[0] == "UF250" or info[0] == "UUF250")):
                    if label == 1:
                        self.sat += int(info[2])
                    else:
                        self.unsat += int(info[2])
                
                for file in os.listdir(current_dir):
                    f = open(current_dir + "/" + file, "r")
                    clauses = f.readlines()[8:]
                    clauses = [line.strip() for line in clauses]
                    clauses = [line[:-2] for line in clauses]
                    clauses = [line for line in clauses if line != '']
                    
                    node_values = [] # idx 2 node values(the frequency)
                    node_freq = []
                    
                    edges_1 = [] # start node idx of each edge
                    edges_2 = [] # end node idx of each edge
                    edge_attr = []
                    
                    
                    for i in range(var_num):
                        tmp = i + var_num
                        edges_1 += [i]
                        edges_1 += [tmp]
                        edges_2 += [tmp]
                        edges_2 += [i]
                        
                        edge_attr += [[1,0]]
                        
                        node_freq += [0]
                    
                    count = 0
                    # build the dictionary
                    for clause in clauses:
                        clause_vars = clause.split(" ")
                        clause_vars = [int(var) for var in clause_vars]
                        
                        for var in clause_vars: 
                            node_freq[abs(var)-1] += 1
                            tmp = [var-1] if var > 0 else [abs(var) + var_num - 1]
                            edges_1 += [count + 2*var_num]
                            edges_1 += tmp
                            edges_2 += tmp
                            edges_2 += [count + 2*var_num]
                            
                            edge_attr += [[0,1]]
                            
                        count += 1
                    """
                    xs = []
                    for i in node_freq:
                        sym = int(random.choice([1,-1]))
                        xi = ((i - 0.5) / 22) * sym
                        xs += [[xi]]
                    node_values = xs
                    node_values += [[-i] for [i] in xs]
                    node_values += [[1] for _ in range(clause_num)]
                    """
                    x_i = [[np.random.uniform(low=-1.0, high=1.0)] for _ in range(0, var_num)]
                    node_values = x_i
                    node_values += [[-i] for [i] in x_i]
                    node_values += [[1] for _ in range(0, clause_num)]
                    
                    f.close()
                    
                    if self.seperate and (info[0] == "UF250" or info[0] == "UUF250"):
                        self.df_test.loc[len(self.df_test)] = [node_values, [edges_1, edges_2], [edge_attr, edge_attr],[label]]
                    else:
                        self.df.loc[len(self.df)] = [node_values, [edges_1, edges_2], [edge_attr, edge_attr],[label]]

        print("Satisfiable cnfs: ", self.sat)
        print("Unsatisfiable cnfs: ", self.unsat)
        sat_ratio = self.sat / (self.sat + self.unsat)
        print("Satisfiable ratio: ", sat_ratio)
        
        if self.seperate:
            self.df_train = self.df
            print("Training set size: ", len(self.df_train))
            print("Testing set size: ", len(self.df_test))
        else:
            self.df_train = self.df.sample(frac=self.frac)
            self.df_test = self.df.drop(self.df_train.index)
            print("Training set size: ", len(self.df_train))
            print("Testing set size: ", len(self.df_test))
            
        print("\nDataProcessing completed.")
        return (self.unsat / self.sat)