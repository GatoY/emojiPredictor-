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
    df=df.dropna(axis=0)
    columnLen = df.columns.shape[0]
    df.columns = [x for x in range(0, columnLen)]
    df = df.drop(0, 1)
    label = df[columnLen - 1].values
    feature = df.ix[:, :columnLen - 2]
    min_max_scaler = preprocessing.MinMaxScaler()
    # feature = min_max_scaler.fit_transform(feature)
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
    train_data = load_data('dataWithRawLabel.csv')
    # dev_data = load_data('dev_most100.csv')
    Trainlabel, Trainfeature = off_feature_extraction(train_data)
    # devlabel, devfeature = off_feature_extraction(dev_data)
    # Trainlabel = np.concatenate((Trainlabel+devlabel),axis=0)
    # Trainfeature=np.concatenate((Trainfeature+devfeature),axis=0)
    # test_data = load_data('test_most100.csv')
    # testlabel, testfeature = off_feature_extraction(test_data)
    clf = off_model_building()
    train_size, train_loss, test_loss = learning_curve(
        clf, Trainfeature, Trainlabel, train_sizes=[np.linspace(0.1, 1, 5)], cv=5, n_jobs=3)
    print('train done')
    train_loss_mean = np.mean(train_loss, axis=1)
    test_loss_mean = np.mean(test_loss, axis=1)
    with open('LearningCurve.txt', 'w') as f:
        f.write(str(train_loss_mean))
        f.write(str(test_loss_mean))
        f.write(str(train_size))
    plt.figure()
    # max_train_index = np.argmax(train_loss_mean)
    # min_train_index = np.argmin(train_loss_mean)
    # max_test_index = np.argmax(test_loss_mean)
    # min_test_index = np.argmin(test_loss_mean)
    plt.plot(train_size, train_loss_mean, 'o-', color='r', label='Train_Scores')
    plt.plot(train_size, test_loss_mean, 'o-', color='g', label="Valid_Scores")
    # plt.plot(train_size[max_train_index], train_loss_mean[max_train_index], 'ks')
    # plt.plot(train_size[min_train_index], train_loss_mean[min_train_index], 'ks')
    # plt.plot(train_size[max_test_index], test_loss_mean[max_test_index], 'ks')
    # plt.plot(train_size[min_test_index], test_loss_mean[min_test_index], 'ks')
    plt.xlabel('train_size')
    plt.ylabel('Loss')
    plt.legend(loc="best")
    plt.savefig("LearningCurve.png")
    plt.show()
    # clf = off_model_building(param())
    # clf = clf.fit(Trainfeature, Trainlabel)
    # label = clf.predict(testfeature)
    # scores = clf.score(testfeature, testlabel)
    # scores = cross_val_score(clf, Trainfeature, Trainlabel)
    # print(scores)
