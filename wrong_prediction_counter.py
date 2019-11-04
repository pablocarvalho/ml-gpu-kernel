#!/usr/bin/env python
import pandas as pd
from pandas import DataFrame
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-i","--input",type=str, help="input file, a CSV separated by semicolons with kernel names, original classification and scikit methods classifications", required=True)

args = parser.parse_args()
inputFile = args.input

origin = pd.read_csv(inputFile,sep=';')

groupedK1names = origin.groupby("k1_name")
groupedK2names = origin.groupby("k2_name")

kernels1 = groupedK1names.groups.keys()
kernels2 = groupedK2names.groups.keys()

kernelsSet = set(kernels1 + kernels2)

falsePositivesList = []
falseNegativesList = []
mistakesSumList = []

for kernel in kernelsSet:
    falsePositives = len (origin[ (  (origin["k1_name"] == kernel) | (origin["k2_name"] == kernel) ) & (origin["classification"] == 0) & (origin["xgb"] == 1) ])
    falseNegatives = len (origin[ (  (origin["k1_name"] == kernel) | (origin["k2_name"] == kernel) ) & (origin["classification"] == 1) & (origin["xgb"] == 0) ])
    mistakesSum = falsePositives + falseNegatives

    falsePositivesList.append(falsePositives)
    falseNegativesList.append(falseNegatives)
    mistakesSumList.append(mistakesSum)    

finalDataframe = DataFrame()
finalDataframe['kernel'] = list(kernelsSet)
finalDataframe['false_positives'] = list(falsePositivesList)
finalDataframe['false_negatives'] = list(falseNegativesList)
finalDataframe['mistakes_sum'] = list(mistakesSumList)

print finalDataframe.sort_values(by=['mistakes_sum'],ascending=False)
