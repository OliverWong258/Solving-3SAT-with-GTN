# -*- coding: utf-8 -*-
import os
import pandas as pd
from tqdm import tqdm

global_node_values = []

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
                    
                    edges_1 = [] # start node idx of each edge
                    edges_2 = [] # end node idx of each edge
                    edge_attr = []
                    
                    node2idx = {} # var node 2 idx
                    idx2value = {} # var idx 2 value
                    count = 0 # node num
                    
                    # build the dictionary
                    for clause in clauses:
                        clause_vars = clause.split(" ")
                        clause_vars = [int(var) for var in clause_vars]
                        
                        for var in clause_vars:
                            if not var in node2idx.keys():
                                node2idx[var] = count
                                idx2value[count] = 1
                                count += 1
                            else:
                                idx2value[node2idx[var]] += 1
                    for i in range(count):
                        node_values.append(idx2value[i])
                    global global_node_values
                    global_node_values += node_values
                    
                    # build edges between var and -var
                    for xi in node2idx.keys():
                        if xi > 0:
                            if (-1*xi) in node2idx.keys():
                                edges_1 += [node2idx[xi]]
                                edges_1 += [node2idx[-1*xi]]
                                edges_2 += [node2idx[-1*xi]]
                                edges_2 += [node2idx[xi]]
                                edge_attr += [[1,0], [1,0]]
                    # build edges between clause and var
                    for clause in clauses:
                        clause_vars = clause.split(" ")
                        clause_vars = [int(var) for var in clause_vars]
                        node_values.append(-1) # -1 is the value for clause node
                        for xi in clause_vars:
                            edges_1 += [count]
                            edges_1 += [node2idx[xi]]
                            edges_2 += [node2idx[xi]]
                            edges_2 += [count]
                            
                            edge_attr += [[0,1],[0,1]] 
                        count += 1
                    
                    f.close()
                    """
                    print("node2idx: ", node2idx)
                    print("node_values", node_values)
                    print("edge_1: ", edges_1)
                    print("edge_2: ", edges_2)
                    print("edge_attr: ", edge_attr)
                    """
                    if self.seperate and (info[0] == "UF250" or info[0] == "UUF250"):
                        self.df_test.loc[len(self.df_test)] = [node_values, [edges_1, edges_2], [edge_attr, edge_attr],[label]]
                    else:
                        self.df.loc[len(self.df)] = [node_values, [edges_1, edges_2], [edge_attr],[label]]

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
    

"""
test = processed_data("./data")
test.process_rawdata()
print(max(global_node_values))
"""