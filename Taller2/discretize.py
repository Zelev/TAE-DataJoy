import pandas as pd
import numpy as np


def read_data(path, perm):
    '''
    This function will get a data file and return
    the labels and data rows separated
    '''
    data_file = open(path, perm)
    raw = [x.split() for x in data_file]
    data_file.close()
    labels = raw[0]  # Create list of labels
    aux = raw[1:] # list of data
    data = []
    for data_line in aux:
        data.append([float(x) for x in data_line])
    
    discretize_data(path, data)
    return labels, data

def discretize_data(path, data):
    data_aux = [x[13] for x in data]
    data_discrete = pd.qcut(data_aux, 3, labels=False)
    for i, item in enumerate(data):
        data[i][13] = data_discrete[i]
        # print item
    return data


file_path = 'Taller2/housing.txt'
labels, data = read_data(file_path, 'r')
raw = [x.strip(',') for x in data] 
print()
'''archi=open('housing_discret2.txt','a')
archi.writelines(labels)
archi.write('\n')
archi.writelines(str(linea).strip('[]').strip(',')+'\n' for linea in data)
archi.close()'''
#print([str(linea).strip('[]') for linea in data])