#This file has NO relation with function
#Solely for myself testing new libraries
import pandas as pd
import torch
import jieba
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
for i in range(0, len(t)):
    print(list(jieba.cut(t["text"][i])))