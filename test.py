#This file has NO relation with function
#Solely for myself testing new libraries
import pandas
import torch
import jieba
import tqdm

test_data = pandas.read_csv('data/SentimentData.csv')
print(test_data.head())
print(test_data.columns)
print(test_data.shape)
print(test_data["label"].value_counts())