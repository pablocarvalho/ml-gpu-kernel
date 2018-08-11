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
from sklearn import metrics 

origin = pandas.read_csv("input_CE.csv", sep=';')
origin = origin.drop(["k1_name"], axis =1 )
origin = origin.drop(["k2_name"], axis =1 )
origin = origin.drop(["classification"], axis =1 )
origin_Y = origin.ce_average
origin = origin.drop(["ce_average"], axis =1 )
origin = DataFrame(scale(origin), index=origin.index, columns=origin.columns)
frames = [origin,origin_Y]
origin = concat(frames,axis=1)
#print origin

trainSet, testSet = train_test_split(origin,train_size = 0.9, test_size = 0.1)

trainSet_X = trainSet.drop(["ce_average"], axis =1 )
trainSet_Y = trainSet.ce_average

testSet_X = testSet.drop(["ce_average"], axis =1 )
testSet_Y = testSet.ce_average
testSet_Y = testSet_Y.astype('int')

lr = linear_model.TheilSenRegressor()
lr.fit(trainSet_X,trainSet_Y)
predicted = lr.predict(testSet_X)
predicted_train = lr.predict(trainSet_X)

print "Multivariate Regression results train:"
print "explained variance: " 	+ str(metrics.explained_variance_score(trainSet_Y,predicted_train)) 
print "mean absolute error: " +	str(metrics.mean_absolute_error(trainSet_Y,predicted_train))
print "mean squared error: " +	str(metrics.mean_squared_error(trainSet_Y,predicted_train)) 	 
print "median absolute error: " +	str(metrics.median_absolute_error(trainSet_Y,predicted_train)) 	 
print "r2: " + str(metrics.r2_score(trainSet_Y,predicted_train))
print "\n"
print "Multivariate Regression results test:"
print "explained variance: " 	+ str(metrics.explained_variance_score(testSet_Y,predicted)) 
print "mean absolute error: " +	str(metrics.mean_absolute_error(testSet_Y,predicted))
print "mean squared error: " +	str(metrics.mean_squared_error(testSet_Y,predicted)) 	 
print "median absolute error: " +	str(metrics.median_absolute_error(testSet_Y,predicted)) 	 
print "r2: " + str(metrics.r2_score(testSet_Y,predicted))
