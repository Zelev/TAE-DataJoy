from sklearn.cross_validation import KFold
from sklearn import linear_model
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

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

def get_min_max(data, data_index):
    data = np.array(data)
    data_min = min(data[:, [data_index]]) - .5
    data_max = max(data[:, [data_index]]) + .5
    return data_min, data_max


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

logreg = linear_model.LogisticRegression(C=1e5)

r = logreg.fit(X_train, Y_train)
z = logreg.predict(X_test)
s = logreg.score(X_test, range(50))
conf_mx = confusion_matrix(z, Y_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test,z,2.)
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
'''
max_min_dict = {}
for index, label in enumerate(housing_labels):
    max_min_dict[label] = {}
    data_min, data_max = get_min_max(housing_data, index)
    max_min_dict[label]['max'] = data_max
    max_min_dict[label]['min'] = data_min
    np_array = (np.arange(data_min, data_max, 0.2))
    #max_min_dict[label]['mesh'] = np.meshgrid(np_array)

max_min_dicte = []
max_min_dicte = np.meshgrid(np_array)

#res_logreg = {}
#for label in housing_labels:
    #evalua = max_min_dict[label]['mesh']
    #rows = len(evalua)
    #cols = len(evalua[0])
    #return [[matriz[j][i] for j in xrange(rows)] for i in xrange(cols)]

#p = [x.ravel() for x in max_min_dict[label]['mesh'][0] for label in housing_labels[0:13]]
#res_logreg = logreg.predict(X)
'''
    