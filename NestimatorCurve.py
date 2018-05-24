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
from sklearn.learning_curve import validation_curve
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
    return np.arange(50, 300, step=50)


if __name__ == "__main__":
    train_data = load_data('train_most100.csv')
    Trainlabel, Trainfeature = off_feature_extraction(train_data)
    # test_data = load_data('test_most100.csv')
    # testlabel, testfeature = off_feature_extraction(test_data)
    param_range = param()
    train_loss, test_loss = validation_curve(
        RandomForestClassifier(random_state=1), Trainfeature, Trainlabel, param_name='n_estimators', param_range=param_range, cv=5)
    print('train done')
    train_loss_mean = np.mean(train_loss, axis=1)
    test_loss_mean = np.mean(test_loss, axis=1)
    with open('record.txt', 'w') as f:
        f.write(str(train_loss_mean))
        f.write(str(test_loss_mean))
        f.write(str(param_range))
    plt.figure()
    # max_train_index = np.argmax(train_loss_mean)
    # min_train_index = np.argmin(train_loss_mean)
    # max_test_index = np.argmax(test_loss_mean)
    # min_test_index = np.argmin(test_loss_mean)
    plt.plot(param_range, train_loss_mean, 'o-', color='r', label='Training')
    plt.plot(param_range, test_loss_mean, 'o-', color='g', label="Cross-validation")
    # plt.plot(param_range[max_train_index], train_loss_mean[max_train_index], 'ks')
    # plt.plot(param_range[min_train_index], train_loss_mean[min_train_index], 'ks')
    # plt.plot(param_range[max_test_index], test_loss_mean[max_test_index], 'ks')
    # plt.plot(param_range[min_test_index], test_loss_mean[min_test_index], 'ks')
    plt.xlabel('n_estimator')
    plt.ylabel('Loss')
    plt.legend(loc="best")
    plt.savefig("Curve_N_estimators.png")
    # clf = off_model_building(param())
    # clf = clf.fit(Trainfeature, Trainlabel)
    # label = clf.predict(testfeature)
    # scores = clf.score(testfeature, testlabel)
    # scores = cross_val_score(clf, Trainfeature, Trainlabel)
    # print(scores)
