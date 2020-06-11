import sklearn
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


def read_data():
    train_feature_name = 'adult_train_feature.txt'
    train_label_name = 'adult_train_label.txt'
    test_feature_name = 'adult_test_feature.txt'
    test_label_name = 'adult_test_label.txt'

    train_feature = pd.read_csv(train_feature_name, sep=" ", header=None)
    train_label = pd.read_csv(train_label_name, sep=" ", header=None)
    test_feature = pd.read_csv(test_feature_name, sep=" ", header=None)
    test_label = pd.read_csv(test_label_name, sep=" ", header=None)

    return train_feature, train_label, test_feature, test_label

def AUC(y, pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
    auc = metrics.auc(fpr, tpr)
    return auc
        
def AdaBoost(X_train, Y_train, X_test, Y_test, M, clf):
    n_train, n_test = len(X_train), len(X_test)
    # Initialize weights
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]
    
    for i in range(M):
        # Fit a classifier with the specific weights
        clf.fit(X_train, Y_train, sample_weight = w)
        pred_train_i = clf.predict(X_train)
        pred_test_i = clf.predict(X_test)
        # Indicator function
        miss = [int(x) for x in (pred_train_i != Y_train)]
        # Equivalent with 1/-1 to update weights
        miss2 = [x if x==1 else -1 for x in miss]
        # Error
        err_m = np.dot(w, miss) / sum(w)
        if err_m > 0.5:
            break
        # Alpha
        alpha_m = 0.5 * np.log( (1 - err_m) / float(err_m))
        # New weights
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        # Add to prediction
        pred_train = [sum(x) for x in zip(pred_train, [x * alpha_m for x in pred_train_i])]
        pred_test = [sum(x) for x in zip(pred_test, [x * alpha_m for x in pred_test_i])]
    
    pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
    return get_error_rate(pred_train, Y_train), get_error_rate(pred_test, Y_test)

def get_error_rate(pred, Y):
    return sum(pred != Y) / float(len(Y))

def main():
    X_train, Y_train, X_test, Y_test = read_data()
    Y_train = Y_train.to_numpy().reshape(1, -1)[0]
    Y_test = Y_test.to_numpy().reshape(1, -1)[0]

    M = 200
    clf_tree = DecisionTreeClassifier(max_depth = 1, random_state = 1)
    error = AdaBoost(X_train, Y_train, X_test, Y_test, 100, clf_tree)
    print(error)

if __name__ == '__main__':
    main()