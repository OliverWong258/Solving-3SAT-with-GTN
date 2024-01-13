# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# ���ݼ�����
dictionary = {"node_features": [],
                      "edges": [],
                      "edge_attr": [],
                      "label": []}

class process_raw():
    def __init__(self, directory = "./data", separate = False, frac=0.8):
        self.df = pd.DataFrame(dictionary)
        self.df_train = pd.DataFrame(dictionary)
        self.df_test = pd.DataFrame(dictionary)
        self.directory = directory
        self.separate = separate
        self.frac = frac
        
    def dataset_processing(self):
        print("Processing raw data")

        num_sat = 0
        num_unsat = 0

        for folder in os.listdir(self.directory):
            current_dir = self.directory + "/" + folder
            if os.path.isdir(current_dir):
                # Ŀ¼=����/������+������+�־���+ʵ������
                info = folder.split(".")
                var_num = int(info[0][2:]) if info[0][1] == "F" else int(info[0][3:])
                clause_num = int(info[1])
                # sat: 1, unsat: 0
                label = 1 if info[0][1] == "F" else 0

                # we want to see the balancing of the training dataset
                if not (self.separate and (info[0] == "UF250" or dir[0] == "UUF250")):
                    if label == 1:
                        num_sat += int(info[2])
                    else:
                        num_unsat += int(info[2])

                
                tmp = [[np.random.uniform(low=-1.0, high=1.0)] for _ in range(0, var_num)]
                node_values = tmp
                node_values += [[-i] for [i] in tmp]    # ����ȡ�Ƕ�Ӧ�Ľڵ�����ȡ����
                node_values += [[1] for _ in range(0, clause_num)]  # �־�ڵ�ͬһ��1��ʾ

                for file in os.listdir(current_dir):
                    cnf = open(current_dir + "/" + file, "r")
                    clauses = cnf.readlines()[8:]
                    clauses = [line.strip() for line in clauses]  
                    clauses = [line[:-2] for line in clauses] 
                    clauses = [line for line in clauses if line != '']

                    # �ߵ���ʼ�ڵ�
                    edges_1 = []
                    edges_2 = []

                    # ������
                    edge_attr = []

                    # �ӱ�����ȡ��֮��ı�
                    for i in range(var_num):
                        edges_1 += [i]
                        edges_1 += [i + var_num]
                        edges_2 += [i + var_num]
                        edges_2 += [i]
                            
                        # [1,0]���������ı�
                        edge_attr += [[1, 0]]

                    cnt = 0
                    for clause in clauses:
                        clause_vars = clause.split(" ")
                        clause_vars = [int(var) for var in clause_vars]
                        # �־�������ı�
                        for var in clause_vars:
                            tmp = [var-1] if var > 0 else [abs(var)-1 + var_num]
                            edges_1 += [cnt + 2*var_num]
                            edges_1 += tmp
                            edges_2 += tmp
                            edges_2 += [cnt + 2*var_num]

                            edge_attr += [[0, 1]]

                        cnt += 1

                    cnf.close()


                    if self.separate and (info[0] == "UF250" or info[0] == "UUF250"):
                        self.df_test.loc[len(self.df_test)] = [node_values, [edges_1, edges_2],
                                                     [edge_attr, edge_attr], [label]]
                    else:
                        self.df.loc[len(self.df)] = [ node_values, [edges_1, edges_2],
                                           [edge_attr, edge_attr], [label]]

        print('Satisfiable: ', num_sat)
        print('Unsatisfiable: ', num_unsat)
        
        pos_weight = num_unsat / num_sat
        print("Positive weitht: ", pos_weight)
        if self.separate_test:
            self.df_train = self.df
            print('Training set size: ', len(self.df))
            print('Test set size: ', len(self.df_test))
        else:
            self.df_train = self.df.sample(frac=self.frac)
            self.df_test = self.df.drop(self.df_tr.index)
            print('Training set size: ', len(self.df_tr))
            print('Test set size: ', len(self.df_test))
        
        print("Raw data processed")

        return pos_weight