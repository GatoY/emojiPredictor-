# logistic, no preprocess.
# top10
from sklearn.model_selection import cross_val_score
import pandas as pd
# from sklearn import cross_validation
# from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
# import logistic
# import svm
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt


def load_data(file_name):
    data = pd.read_csv(file_name)
    return data


def off_feature_extraction(df):
    columnLen = df.columns.shape[0]
    df.columns = [x for x in range(0, columnLen)]
    df = df.drop(0, 1)
    label = df[columnLen - 1].values
    feature = df.ix[:, :columnLen - 2]
    min_max_scaler = preprocessing.MinMaxScaler()
    feature = min_max_scaler.fit_transform(feature)
    return label, feature


def featureEng(df, indices):
    columnLen = df.columns.shape[0]
    df.columns = [x for x in range(0, columnLen)]
    df = df.drop(0, 1)
    for i in indices:
        # print(i)
        df = df.drop(i + 1, 1)
    # print(df.head())

    label = df[columnLen - 1].values
    df.columns = [x for x in range(0, df.columns.shape[0])]
    feature = df.ix[:, :df.columns.shape[0] - 2]
    print(feature)
    min_max_scaler = preprocessing.MinMaxScaler()
    feature = min_max_scaler.fit_transform(feature)
    return label, feature


def off_model_building():
    # n_estimator 80-300 min_samples_split 2-4 min_samples_leaf 1-2
    clf = RandomForestClassifier(bootstrap=False, n_estimators=200, criterion='gini', max_depth=None, max_features=3,
                                 min_samples_leaf=2, min_samples_split=9)

    return clf


def test_feature_extraction(df):
    feature = df.ix[:].values
    return feature


def param():
    # n_estimators
    return np.arange(80, 300, step=10)


if __name__ == "__main__":
    train_data = load_data('train_most100.csv')
    dev = load_data('dev_most100.csv')
    columnLen = train_data.columns.shape[0]
    train_data.columns = [x for x in range(0, columnLen)]
    columnLen = dev.columns.shape[0]
    dev.columns = [x for x in range(0, columnLen)]

    frames = [train_data, dev]
    train_data = pd.concat(frames)
    Trainlabel, Trainfeature = off_feature_extraction(train_data)
    clf = off_model_building()
    clf = clf.fit(Trainfeature, Trainlabel)
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    with open('most100.txt') as f:
        line = f.readlines()
    with open('recordimportance.txt','w') as f:
        for i in range(0,100):
            f.write(str(importances[indices[i]]))
            f.write(line[indices[i]])
            f.write('\n')
