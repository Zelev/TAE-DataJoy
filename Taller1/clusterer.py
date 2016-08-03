from diabetes_TAE import *
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation
import itertools

data, labels, measures = giveme_data()
data_notest = [x[:8] for x in data ]#if 'tested_positive' not in x[8]]

xaxis = 6
yaxis = 1
X = [float(x[xaxis]) for x in data_notest]
Y = [float(x[yaxis]) for x in data_notest]
data_show = [[X[i], Y[i]] for i in xrange(len(data_notest))]
cluster_plot = KMeans(n_clusters=10).fit_predict(data_show)
#cluster_plot = DBSCAN(eps=1.0).fit_predict(data_show)
#cluster_plot = AffinityPropagation(damping=0.9).fit_predict(data_show)
plt.scatter(X, Y, c=cluster_plot)
plt.xlabel(labels[xaxis])
plt.ylabel(labels[yaxis])
plt.show()


data_whit = np.array(data_notest).astype(float)
data_whit = whiten(data_show)

clusters = {'diffs':[0], 'dist':[]}

for clus_num in xrange(15):
    clus_num += 1
    centers, distortion = kmeans(data_whit, clus_num)
    clusters['dist'].append(distortion)
    if clus_num > 1:
        clusters['diffs'].append(clusters['dist'][clus_num - 2] - clusters['dist'][clus_num - 1])
    
print clusters
x_axis = range(16)[1:]
plt.title("Punto 2: Busqueda del mejor numero de clusters")
vals, = plt.plot(x_axis,clusters['dist'], '>', label='WhitinSS')
diffs, =plt.plot(x_axis,clusters['diffs'],label='difference')
plt.legend(handles=[vals, diffs])
plt.xlabel('Numero de Clusters')
plt.show()

