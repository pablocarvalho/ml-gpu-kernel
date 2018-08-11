from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from pandas import DataFrame
import pandas
from math import sqrt
from sklearn import metrics 
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt 


X = pandas.read_csv("input_throughput.csv", sep=';')
X = X.drop(["k1_name"], axis =1 )
X = X.drop(["k2_name"], axis =1 )
X = X.drop(["classification"], axis =1 )
Y = X.throughput
X = X.drop(["throughput"], axis =1 )
newdf = DataFrame(scale(X), index=X.index, columns=X.columns)


lr = linear_model.LinearRegression()
predicted = cross_val_predict(lr, newdf, Y, cv=KFold(10),n_jobs = 6)
scores = cross_val_score(lr, newdf, Y,  cv=KFold(10))
print scores
print "\n"

k_fold = KFold(n_splits=10)
split_result = k_fold.split(newdf)

stage = 1;
for train_indices, test_indices in split_result:
    lr.fit(newdf.iloc[train_indices],Y.iloc[train_indices])
    score = lr.score(newdf.iloc[test_indices],Y.iloc[test_indices])
    print "stage " +str(stage) +" score: " + str(score)
    stage+=1
print "\n"


variables = list(X.columns.values)


for i in range(len(variables)):
	print variables[i] + "\t\t\t\t" + str(lr.coef_[i])
print "\n"

#acum = 0
#for i in range(len(Y)):
#    dist=sqrt((Y[i]-predicted[i])*(Y[i]-predicted[i]))
#    acum=acum+dist

#mean=acum/len(Y)


print "explained variance: " 	+ str(metrics.explained_variance_score(Y,predicted)) 
print "mean absolute error: " +	str(metrics.mean_absolute_error(Y,predicted))
print "mean squared error: " +	str(metrics.mean_squared_error(Y,predicted)) 	 
print "median absolute error: " +	str(metrics.median_absolute_error(Y,predicted)) 	 
print "r2: " + str(metrics.r2_score(Y,predicted))

fig, ax = plt.subplots()
ax.scatter(Y, predicted, edgecolors=(0, 0, 0)) 
ax.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

