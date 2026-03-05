#This file has NO relation with function
#Solely for myself testing new libraries
import pandas as pd
import torch
import jieba as jb
import tqdm

def PrintCSV() -> None:
    test_data = pandas.read_csv('data/SentimentData.csv')
    print(test_data.head())
    print(test_data.columns)
    print(test_data.shape)
    print(test_data["label"].value_counts())

def CutVocabulary() -> None:
    test_data = pandas.read_csv('data/SentimentData.csv')
    print(test_data.loc[1:3, "text":"label"])

t = pd.read_csv('data/SentimentData.csv')
for i in t["text"]:
    print(list(jb.cut(str(i))))