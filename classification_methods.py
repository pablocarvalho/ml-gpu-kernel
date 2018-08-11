from sklearn.neural_network import MLPClassifier
import pandas
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score, precision_score, recall_score
from pandas import DataFrame
from pandas import concat
import numpy as np
from numpy import split
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

origin = pandas.read_csv("input_CE.csv", sep=';')
origin = origin.drop(["k1_name"],axis = 1)
origin = origin.drop(["k2_name"],axis = 1)
origin_Y = origin.classification
origin = origin.drop(["classification"], axis =1 )
origin = DataFrame(scale(origin), index=origin.index, columns=origin.columns)
frames = [origin,origin_Y]
origin = concat(frames,axis=1)
print origin

#trainSet, validationSet, testSet = split(origin.sample(frac=1), [int(.7*len(origin)), int(.9*len(origin))])
#trainSet, testSet = split(origin.sample(frac=1), [int(.9*len(origin))])
#trainSet, testSet = split(origin, [int(.9*len(origin))])
trainSet, testSet = train_test_split(origin,train_size = 0.9, test_size = 0.1, stratify = origin.classification)

trainSet_X = trainSet.drop(["classification"], axis =1 )
trainSet_Y = trainSet.classification
#trainSet_Y=trainSet_Y.astype('int')

#validationSet_X = validationSet.drop(["classification"], axis =1 )
#validationSet_Y = validationSet.classification

testSet_X = testSet.drop(["classification"], axis =1 )
testSet_Y = testSet.classification
testSet_Y = testSet_Y.astype('int')


clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, early_stopping = True, validation_fraction = 0.22)
clf.fit(trainSet_X, trainSet_Y) 

mlp_prediction = clf.predict(testSet_X)

print "\n"
print "Multi-layer Perceptron results:"
print "accuracy_score: " + str(accuracy_score(testSet_Y,mlp_prediction))
print "precision_score :" + str(precision_score(testSet_Y,mlp_prediction,average='micro'))
print "recall_score: "+ str(recall_score(testSet_Y,mlp_prediction,average='micro'))
print "\n"



knn = KNeighborsClassifier()
knn.fit(trainSet_X, trainSet_Y)
knn_prediction = knn.predict(testSet_X)

print "KNN results:"
print "accuracy_score: " + str(accuracy_score(testSet_Y,knn_prediction))
print "precision_score :" + str(precision_score(testSet_Y,knn_prediction,average='micro'))
print "recall_score: "+ str(recall_score(testSet_Y,knn_prediction,average='micro'))
print "\n"


logistic = linear_model.LogisticRegression()
logistic.fit(trainSet_X, trainSet_Y)
logistic_prediction = logistic.predict(testSet_X)

print "Logistic Regression results:"
print "accuracy_score: " + str(accuracy_score(testSet_Y,logistic_prediction))
print "precision_score :" + str(precision_score(testSet_Y,logistic_prediction,average='micro'))
print "recall_score: "+ str(recall_score(testSet_Y,logistic_prediction,average='micro'))
print "\n"
