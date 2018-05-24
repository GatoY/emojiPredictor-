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


def reflect(filename):
    with open(filename) as f:
        tokens = [x[:-1] for x in f.readlines()]


def off_feature_extraction(df):
    columnLen = df.columns.shape[0]
    df.columns = [x for x in range(0, columnLen)]
    df = df.drop(0, 1)
    label = df[columnLen - 1].values
    feature = df.ix[:, :columnLen - 2]
    min_max_scaler = preprocessing.MinMaxScaler()
    feature = min_max_scaler.fit_transform(feature)
    return label, feature


def off_model_building(estimators):
    # n_estimator 80-300 min_samples_split 2-4 min_samples_leaf 1-2
    clf = RandomForestClassifier(random_state=1, n_estimators=estimators)
    return clf


def test_feature_extraction(df):
    feature = df.ix[:].values
    return feature


def param():
    # n_estimators
    return np.arange(80, 300, step=10)


if __name__ == "__main__":
    train_data = load_data('train_most100.csv')
    Trainlabel, Trainfeature = off_feature_extraction(train_data)
    # test_data = load_data('test_most100.csv')
    # testlabel, testfeature = off_feature_extraction(test_data)
    clf = off_model_building(200)
    clf = clf.fit(Trainfeature, Trainlabel)
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(Trainfeature.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, Trainlabel[f], importances[indices[f]]))
    # label = clf.predict(testfeature)
    # scores = clf.score(testfeature, testlabel)
    # scores = cross_val_score(clf, Trainfeature, Trainlabel)
    # print(scores)
