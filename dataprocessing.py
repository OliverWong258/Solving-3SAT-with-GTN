# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# 数据集列名
dictionary = {"numberOfVariables": [],
                      "numberOfClauses": [],
                      "variablesSymb": [],
                      "variablesNum": [],
                      "edges": [],
                      "edgeAttr": [],
                      "label": []}

class process_raw():
    def __init__(self, directory = "./data", separate_test = False, frac=0.8):
        self.df = pd.DataFrame(dictionary)
        self.df_test = pd.DataFrame(dictionary)
        self.directory = directory
        self.separate_test = separate_test
        self.frac = frac
        
    def dataset_processing(self):
        print("Start the data processing...\n")

        satisfiable_num = 0
        unsatisfiable_num = 0

        for dirName in os.listdir(self.directory):
            curr_dir = self.directory + "/" + dirName
            if os.path.isdir(curr_dir):
                # the directory's name stored information about its contents :[UF/UUF<#variables>.<#clauses>.<#cnfs>]
                dir_info = dirName.split(".")
                # number of variables in each data-file regarding the folder
                number_of_variables = int((re.findall(r'\d+', dir_info[0]))[0])
                # number of clauses in each
                # data-file regarding the folder
                number_of_clauses = int(dir_info[1])
                # get label of these data : UUF means UNSAT and UF means SAT
                y = 0 if dir_info[0][:3] == "UUF" else 1

                # we want to see the balancing of the training dataset
                if not (self.separate_test and (dir_info[0] == "UF250" or dir_info[0] == "UUF250")):
                    if y == 1:
                        satisfiable_num += int(dir_info[2])
                    else:
                        unsatisfiable_num += int(dir_info[2])

                # Nodes:
                #     0 - numberOfVariables- 1                                      : x_1 - x_n
                #     numberOfVariables - 2*numberOfVariables                       : ~x_1 - ~x_n
                #     2*numberOfVariables - 2*numberOfVariables + numberOfClauses   : c_1 - c_m

                nodes = [i for i in range(0, 2 * number_of_variables + number_of_clauses)]
                x_i = [[np.random.uniform(low=-1.0, high=1.0)] for _ in range(0, number_of_variables)]
                node_values = x_i
                node_values += [[-i] for [i] in x_i]
                node_values += [[1] for _ in range(0, number_of_clauses)]

                for fileName in os.listdir(curr_dir):
                    f = open(curr_dir + "/" + fileName, "r")
                    clauses = f.readlines()[8:]
                    clauses = [line.strip() for line in clauses]  # remove '\n' from the end and '' from the start
                    clauses = [line[:-2] for line in clauses]     # keep only the corresponding variables
                    clauses = [line for line in clauses if line != '']  # keep only the lines that correspond to a clause

                    # edges
                    edges_1 = []
                    edges_2 = []

                    # compute edge attributes as x_i -> ~x_i are
                    # connected via a different edge than c_j and x_i
                    edge_attr = []

                    # make the edges from x_i -> ~x_i and ~x_i -> x_i
                    for i in range(number_of_variables):

                        temp = [i + number_of_variables]
                        edges_1 += [i]
                        edges_1 += temp
                        edges_2 += temp
                        edges_2 += [i]

                        # first characteristic is :  connection between x_i and ~x_i
                        # second characteristic is :  connection between c_j and x_i
                        edge_attr += [[1, 0]]

                    # make the edges from corresponding c_j -> x_i (NOW VICE VERSA)
                    count = 0
                    for clause in clauses:
                        clause_vars = clause.split(" ")
                        clause_vars = [int(var) for var in clause_vars]
                        # create the corresponding edges
                        for xi in clause_vars:
                            temp = [xi-1] if xi > 0 else [abs(xi)-1+number_of_variables]
                            edges_1 += [count + 2*number_of_variables]
                            edges_1 += temp
                            edges_2 += temp
                            edges_2 += [count + 2*number_of_variables]

                            edge_attr += [[0, 1]]

                        count += 1

                    f.close()

                    # insert new row in dataframe :
                    # "numberOfVariables","numberOfClauses", "variablesSymb", "variablesNum", "edges", "edgeAttr","label"
                    if self.separate_test and (dir_info[0] == "UF250" or dir_info[0] == "UUF250"):
                        self.df_test.loc[len(self.df_test)] = [number_of_variables, number_of_clauses,
                                                     node_values, nodes, [edges_1, edges_2],
                                                     [edge_attr, edge_attr], [y]]
                    else:
                        self.df.loc[len(self.df)] = [number_of_variables, number_of_clauses,
                                           node_values, nodes, [edges_1, edges_2],
                                           [edge_attr, edge_attr], [y]]

        # print some metrics
        print(f'Satisfiable CNFs   : {satisfiable_num}')
        print(f'Unsatisfiable CNFs : {unsatisfiable_num}\n')

        sat_ratio = satisfiable_num / (satisfiable_num + unsatisfiable_num)

        print(f'Ratio of SAT   : {sat_ratio:.4f}')
        print(f'Ratio of UNSAT : {1.0 - sat_ratio:.4f}\n')
        
        if self.separate_test:
            print(f'Training set size: {len(self.df)}')
            print(f'Test set size: {len(self.df_test)}')
        else:
            self.df_tr = self.df.sample(frac=self.frac)
            self.df_test = self.df.drop(self.df_tr.index)
            print(f'Training set size: {len(self.df_tr)}')
            print(f'Test set size: {len(self.df_test)}')

        print("\nProcessing completed.")

        # return this for later purposes
        return unsatisfiable_num/satisfiable_num