from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus


def load_data():
    test_data = pd.read_csv('test.csv')
    train_data = pd.read_csv('train.csv')
    
    train_features = np.array(train_data[train_data.columns.drop('target')])
    train_targets = np.array(train_data['target'])
    test_features = np.array(test_data[test_data.columns.drop('target')])
    test_targets = np.array(test_data['target'])

    return train_features, train_targets, test_features, test_targets

def tree_depth_optimization(features, targets, range):

    hyperparameter_range = np.arange(range[0], range[1])
    
    accuracies = []

    for param in hyperparameter_range:
        model = tree.DecisionTreeClassifier(max_depth = param)
        accuracies.append(kfold_average_accuracy(10, features, targets, model))

    best_hyperparameter = hyperparameter_range[accuracies.index(max(accuracies))]

    print("Hyperparameters: {0}".format(hyperparameter_range))
    print("Hyperparameter Accuracies: {0}".format(accuracies))
        
    return best_hyperparameter

def kfold_average_accuracy(n_folds, features, targets, model):
    
    kf = KFold(n_splits = n_folds)
    validation_accuracies = {}

    i = 0

    for train_index, test_index in kf.split(features, targets):
        
        x_train = features[train_index]
        y_train = targets[train_index]

        x_validation = features[test_index]
        y_validation = targets[test_index]

        model.fit(x_train, y_train)
        validation_accuracies[i] = model.score(x_validation, y_validation)
        i += 1
    
    average_test_accuracy = sum(validation_accuracies.values())/n_folds

    print("Average Validation Accuracy: {0}".format(average_test_accuracy))

    return average_test_accuracy

def main():
    train_features, train_targets, test_features, test_targets = load_data()

    
    original_data = pd.read_csv('train.csv')

    best_depth = tree_depth_optimization(train_features, train_targets, [2,12])

    model = tree.DecisionTreeClassifier(max_depth = best_depth)
    model.fit(train_features, train_targets)

    train_accuracy = model.score(train_features, train_targets)
    test_accuracy = model.score(test_features, test_targets)
    print('Best Depth: {0}'.format(best_depth))
    print("Train Accuracy: {0} \n Test Accuracy: {1}".format(train_accuracy, test_accuracy))
    
    print('test_target_sum: {0}'.format(sum(train_targets)))
    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data,  
                filled=True, rounded=True, class_names = ['0', '1'],
                special_characters=True, feature_names=original_data.columns.drop('target'))
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    Image(graph.create_png())
    graph.write_png('decision_tree.png')

    return

if __name__ == "__main__":
    main()