#Author : Deepansh Dubey.
#Date   : 10/10/2021.

#loading required modules
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

#loading datasets
iris = datasets.load_iris()

#printing features & labels
print(iris.DESCR)
features = iris.data
labels = iris.target
print(features[0], labels[0])

#training the classifier
clf=KNeighborsClassifier()
clf.fit(features, labels)

preds = clf.predict([[1,1,1,1]])
print(preds)
