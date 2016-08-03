from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
import numpy as np

'''
Ejemplo de clase pero hecho en python porque somos rudos y rebeldes xD
'''
def create_matrix():
    '''
    Generates a Matrix from the data.txt file for the dendogram
    to graph
    '''
    
    data_by_name = []
    labels = []
    data_file = open('data.txt', 'r')
    for line in data_file:
        data = line.split()
        labels.append(data[7])
        line = [x for x in data[:7]]
        data_by_name.append(line)
    return np.array(data_by_name).astype(np.float), labels


def graph_dendogram(cluster, x_labels):
    '''graph the actual dendogram'''
    
    plt.figure()
    plt.title('Ejemplo TAE')
    plt.ylabel('distance')
    dendrogram(
        cluster,
        leaf_rotation=90,  # rotates the x axis labels
        leaf_font_size=8,  # font size for the x axis labels
    )
    plt.show()

matrix, labels = create_matrix()

graph_dendogram(matrix, labels)


# info, coph_dist = cophenet(cluster, pdist(matrix[1:]))
# print info
