import math
import numpy
from matplotlib import pyplot as plt
from matplotlib.pylab import hist, show
from matplotlib.ticker import FuncFormatter
from re import findall

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


#Function that recovery and put format to the data on diabetes.txt
#diabetes.txt is a database of assesed people for prove their diabetes state

def giveme_data():
    diabetes_file = open('Taller1/diabetes.txt', 'r')
    data_lines = [x.split(',') for x in diabetes_file]
    
    data_labels = []
    for label in data_lines[44:53]:
        final = len(label[0]) - 1
        data_labels.append(label[0][8:final])
    
    data_measures = []
    for measure in data_lines[66:74]:
        final = len(measure[0])-1
        measures = [float(x) for x in findall("\d+\.\d+", measure[0])]
        measures.append(math.sqrt(measures[1]))
        data_measures.append(measures)

    diab_data = []
    for line in data_lines[95:]:
        diab_data.append([x if x != '0' else None for x in line ])
    
    return diab_data, data_labels, data_measures
    
def counter(data_to_count, label):
    dict_result = {label: {}}
    # dict_reult[label] = {}
    for data in data_to_count:
        try:
            dict_result[label][data] += 1
        except:
            dict_result[label][data] = 1
    return dict_result

def plot_histogram(data, num_val):
    histo = [[float(x[num_val]) for x in data if float(x[num_val]) != 0 if "tested_positive" in x[8]],
                [float(x[num_val]) for x in data if float(x[num_val]) != 0 if "tested_positive" not in x[8]]]
                    
                    
    plt.hist(histo)
    show()

def plot_boxplot(data,num):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.set_window_title('A Boxplot')
    plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
    
    bp = plt.boxplot([float(x[num]) for x in data if float(x[num]) != 0 if "tested_positive" in x[8]], notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.set_window_title('A Boxplot')
    plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
    
    bp = plt.boxplot([float(x[num]) for x in data if float(x[num]) != 0 if "tested_positive" not in x[8]], notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')
    


def mediana(data,num):
    dict_result = {}
    dat = [x[num] for x in data]
    for x in dat:
        try:
            dict_result[x] += 1
        except:
            dict_result[x] = 1
    suma = 0
    vals = []
    for x in dict_result:
        vals.append( dict_result[x])
        
     
    return float(sum(vals)/2)
    


'''def moda(data,num):
    dict_result = {}
    dat = [x[num] for x in data]
    for x in dat:
        try:
            dict_result[x] += 1
        except:
            dict_result[x] = 1
    suma = 0
    vals = []
    for x in dict_result:
        vals.append( dict_result[x])
        
     
    return max(vals)/2)
'''

data, labels, measures = giveme_data()


num_pregnant        = counter(data_to_count=[x[0] for x in data], label=labels[0])
plasma              = counter([x[1] for x in data], labels[1])
diastolic           = counter([x[2] for x in data], labels[2])
tricepSkin          = counter([x[3] for x in data], labels[3])
insuline            = counter([x[4] for x in data], labels[4])
imc                 = counter([x[5] for x in data], labels[5])
diabetes_Pedigree   = counter([x[6] for x in data], labels[6])
age                 = counter([x[7] for x in data], labels[7])
class_variable      = counter([x[8] for x in data], labels[8])


# print(measures)
# print plot_histogram(data,1)

