# logistic, no preprocess.
#top10

import pandas as pd

labelList = {'Clap': [1,0,0,1], 'Hands': [1,0,0,0], 'Upside': [0,1,1,1], 'Think': [0,1,1,0], 'Neutral': [0,1,0,1], 'Shrug': [0,1,0,0], 'FacePalm': [0,0,1,1],
             'Cry': [0,0,1,0], 'Explode': [0,0,0,1], 'Disappoint': [0,0,0,0]}

def preprocess():
    # topList = reflect('top10.txt')
    # mostList = reflect('most100.txt')
    left_data = load_data('train_top10.csv')
    right_data = load_data('train_most100.csv')
    # left_data.drop()
    # columnLen = left_data.columns.shape[0]
    # print(columnLen)
    columnLen = left_data.columns.shape[0]
    left_data.columns = [x for x in range(0, columnLen)]
    left_data=left_data.drop(0,1)
    columnLen = right_data.columns.shape[0]
    # print(columnLen)
    right_data.columns = [x for x in range(0, columnLen)]
    right_data=right_data.drop(columnLen-1,1)
    train_data = pd.concat([right_data, left_data], axis = 1, join='inner')
    columnLen=train_data.columns.shape[0]
    train_data.columns = [x for x in range(0, columnLen)]
    # print(train_data.head())
    # rawLabel = train_data[train_data.columns.shape[0]-1]
    # train_data[train_data.columns.shape[0] - 1]=list(map(lambda x: labelList[x], rawLabel))
    # return train_data
    # train_data.drop(0)
    train_data.to_csv('dataWithRawLabel.csv', index=False, sep=',')

def load_data(file_name):
    data = pd.read_csv(file_name)
    return data

def reflect(filename):
    with open(filename) as f:
        tokens=[x[:-1] for x in f.readlines()]

if __name__ == "__main__":
    preprocess()
