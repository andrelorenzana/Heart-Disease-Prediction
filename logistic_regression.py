import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut

def load_data():
    return pd.read_csv('heart.csv')

def permute_array(array, test_size):

    permuted_data = np.random.permutation(array)
    train_df, test_df = train_test_split(permuted_data, test_size=test_size)
    
    print(np.shape(train_df))

    x_train = train_df[:,:-1]
    y_train = train_df[:,-1]
    

    x_test = test_df[:,:-1]
    y_test = test_df[:,-1]

    return x_train, y_train, x_test, y_test

def average_base_accuracy(n, test_size):
    original_data = load_data()
    train_accuracies = {}
    test_accuracies = {}
    model = LogisticRegression()

    for i in range(n):
        
        x_train, y_train, x_test, y_test = permute_array(original_data, test_size)

        model.fit(x_train, y_train)

        train_accuracies[i] = model.score(x_train, y_train)
        test_accuracies[i] = model.score(x_test, y_test)
    
    print(train_accuracies)
    print(test_accuracies)
    print("Average Train Accuracy: {0}".format(sum(train_accuracies.values())/n))
    print("Average Test Accuracy: {0}".format(sum(test_accuracies.values())/n))
    return

def kfold_average_accuracy(n_folds, features, targets, model):
    
    kf = KFold(n_splits = n_folds)
    train_accuracies = {}
    test_accuracies = {}

    i = 0

    for train_index, test_index in kf.split(features, targets):
        
        x_train = features[train_index]
        y_train = targets[train_index]

        x_test = features[test_index]
        y_test = targets[test_index]

        model.fit(x_train, y_train)
        train_accuracies[i] = model.score(x_train, y_train)
        test_accuracies[i] = model.score(x_test, y_test)
        i += 1
    
    average_train_accuracy = sum(train_accuracies.values())/n_folds
    average_test_accuracy = sum(test_accuracies.values())/n_folds

    print("Average Train Accuracy: {0}".format(average_train_accuracy))
    print("Average Test Accuracy: {0}".format(average_test_accuracy))

    return average_test_accuracy

def regularization_optimization(train_data):

    features = np.array(train_data[train_data.columns.drop('target')])
    targets = np.array(train_data['target'])

    hyperparameter_range = np.linspace(0.1,0.35,10)
    
    accuracies = []

    for i in range(10):
        model = LogisticRegression(penalty='l2', C = hyperparameter_range[i])
        accuracies.append(kfold_average_accuracy(10, features, targets, model))

    best_hyperparameter = hyperparameter_range[accuracies.index(max(accuracies))]

    print("Hyperparameters: {0}".format(hyperparameter_range))
    print("Hyperparameter Accuracies: {0}".format(accuracies))
        
    return best_hyperparameter
    
def logistic_regression_hyperparameter_optimization():
    test_data = pd.read_csv('test.csv')
    train_data = pd.read_csv('train.csv')

    best_c = regularization_optimization(train_data)

    model = LogisticRegression(penalty='l2', C = best_c)

    train_features = np.array(train_data[train_data.columns.drop('target')])
    train_targets = np.array(train_data['target'])
    test_features = np.array(test_data[test_data.columns.drop('target')])
    test_targets = np.array(test_data['target'])

    model.fit(train_features, train_targets)
    training_accuracy = model.score(train_features, train_targets)
    test_accuracy = model.score(test_features, test_targets)

    print('best C: {0}'.format(best_c))
    print("Train Accuracy: {0} \n Test Accuracy: {1}".format(training_accuracy, test_accuracy))

    return


def main():
    logistic_regression_hyperparameter_optimization()
    


if __name__ == '__main__':
    main()