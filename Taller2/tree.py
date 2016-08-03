# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from sklearn import datasets, linear_model
from sklearn.metrics import confusion_matrix


def read_data(path, perm):
    '''
    This function will get a data file and return
    the labels and data rows separated
    '''
    data_file = open(path, perm)
    raw = [x.split(',') for x in data_file]
    labels = raw[0][0].split()  # Create list of labels
    aux = raw[1:] # list of data
    data = []
    for data_line in aux:
        line = [float(x) for x in data_line]
        # line.append(str(data_line[13]))
        data.append(line)
    return labels, data

file_path = 'Taller2/housing_discret2.txt'
housing_labels, housing_data = read_data(file_path, 'r')
result = [x[13] for x in housing_data]

n = len(housing_data)

kf = KFold(n, n_folds=10, shuffle=True,random_state=None)

for train_index, test_index in kf:
    X_train = [housing_data[x] for x in train_index]
    X_test = [housing_data[x] for x in test_index]
    Y_train = [result[x] for x in train_index]
    Y_test = [result[x] for x in test_index]

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X_train, Y_train)
regr_2.fit(X_train, Y_train)

# Predic
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

conf_mx_1 = confusion_matrix(y_1, Y_test)
conf_mx_2 = confusion_matrix(y_2, Y_test)