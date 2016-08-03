import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
import numpy as np
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
    X_result = [housing_data[x][13] for x in test_index]
    X_test = [housing_data[x] for x in test_index]
    Y_train = [result[x] for x in train_index]
    Y_test = [result[x] for x in test_index]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, Y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(X_test) - Y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, Y_test))

x_round = [float('%.1f' % x ) for x in regr.predict(X_test)]
conf_mx = confusion_matrix(x_round, Y_test)