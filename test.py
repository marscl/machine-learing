import KNN.kNN as knn
import matplotlib
import matplotlib.pyplot as plt
from numpy import *

# group, labels = knn.createDataSet()
# clazz=knn.classity0([0,0],group,labels,3)
dataset, labels = knn.file2matrix("datas/dataset.txt")
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataset[:, 0], dataset[:, 1],
           15.0*array(labels),  15.0*array(labels))
#plt.show()

knn.classifyPerson()
