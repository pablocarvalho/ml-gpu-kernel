#!/usr/bin/env python

from sklearn.neural_network import MLPClassifier
import pandas
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score, precision_score, recall_score, cohen_kappa_score
from pandas import DataFrame
from pandas import concat
import numpy as np
from numpy import split
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
import numpy
import tabulate
from scipy.stats import ttest_ind
import argparse
import os
from sklearn.metrics import precision_recall_curve
from funcsigs import signature

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

from xgboost import plot_tree

parser = argparse.ArgumentParser()
parser.add_argument("-m","--mode", type=str, choices=['ce','attempts'], help="mode to execute, ce stands for interference and attempts to concurrency", required=True)
parser.add_argument("-s","--seed", type=int, help="seed used for randomizing folds creation. Original seed from paper will be used if not set")
parser.add_argument("-i","--input",type=str, help="input file, a CSV separated by semicolons with last column composed by the expected classification", required=True)
parser.add_argument("-o","--output-folds", type=str,help="outputs a file for each run (RFE, ALL_VARS, USER_VARS) that contains the classification value for each used classifier")
parser.add_argument("-v","--verbose", help="prints detailed accuracy, precision, recall and kappa score for each fold",action='store_true')
parser.add_argument("-g","--grid-search", help="applies grid-search to the classifiers",action='store_true')
parser.add_argument("-f","--features", type=int, help="number of features to use. If not set, it will use 4 and 6 for attempts and ce modes respectivelly")
parser.add_argument("-p","--precision-recall", type=str, help="input a path to precision-recall plots to be saved")
parser.add_argument("-c","--custom-variables", nargs='*', help="Additional step running with a set of user selected variables. Ex: k1_shared_mem k2_shared_mem k1_registers k2_registers")

args = parser.parse_args()

inputFile = args.input
origin = pandas.read_csv(inputFile,sep=';')
mode = args.mode
seed = args.seed
outputFolds = args.output_folds
features = args.features
verbose = False
gridSearch= False
precisionRecallPath = args.precision_recall
userVariables = []
gridSearchStr=""
namesAndClassification = DataFrame()

userVariables = args.custom_variables

if(args.verbose):
	verbose = True

if(args.grid_search):
	gridSearch=True
	gridSearchStr="GRID_SEARCH_"

if (mode == 'ce' and seed is None):
	seed = 929416
	features = 6

if (mode == 'attempts' and seed is None):
	seed = 204651
	features = 4


class Results:

	def __init__(self, size):
		self.lastPos = 0
		self.accuracies = numpy.zeros(size)
		self.precisions = numpy.zeros(size)
		self.recalls = numpy.zeros(size)
		self.kappas = numpy.zeros(size)
		
	def addResult(self, accuracy,precision,recalls,kappas):

		self.accuracies[self.lastPos] = accuracy
		self.precisions[self.lastPos] = precision
		self.recalls[self.lastPos] = recalls
		self.kappas[self.lastPos] = kappas
		

		self.lastPos+=1

	def printStatistics(self):


		accuracy_mean = numpy.mean(self.accuracies)
		precision_mean = numpy.mean(self.precisions)
		recalls_mean = numpy.mean(self.recalls)
		kappas_mean = numpy.mean(self.kappas)
		mean_array = np.array(['mean',accuracy_mean,precision_mean,recalls_mean,kappas_mean])

		accuracy_std = numpy.std(self.accuracies)
		precision_std = numpy.std(self.precisions)
		recalls_std = numpy.std(self.recalls)
		kappas_std = numpy.std(self.kappas)
		std_array = np.array(['standard dev',accuracy_std,precision_std,recalls_std,kappas_std])

		accuracy_last = self.accuracies[self.lastPos-1]
		precision_last = self.precisions[self.lastPos-1]
		recall_last = self.recalls[self.lastPos-1]
		kappas_last = self.kappas[self.lastPos-1]
		last_array = np.array(['last val',accuracy_last,precision_last,recall_last,kappas_last])

		accuracy_maxValue = numpy.amax(self.accuracies)
		precision_maxValue = numpy.amax(self.precisions)
		recall_maxValue = numpy.amax(self.recalls)
		kappas_maxValue = numpy.amax(self.kappas)
		maxValue_array = np.array(['max val',accuracy_maxValue,precision_maxValue,recall_maxValue,kappas_maxValue])

		accuracy_minValue = numpy.amin(self.accuracies)
		precision_minValue = numpy.amin(self.precisions)
		recall_minValue = numpy.amin(self.recalls)
		kappas_minValue = numpy.amin(self.kappas)
		minValue_array = np.array(['min val',accuracy_minValue,precision_minValue,recall_minValue,kappas_minValue])

		head = np.array(['accuracy','precision','recall','kappa'])		
		table = np.array([mean_array,std_array,last_array,minValue_array, maxValue_array])

		print tabulate.tabulate(table,headers=head)


def plot_confusion_matrix(cm, class_names,filename, title='Confusion matrix', cmap=plt.cm.Blues):
	plt.clf()
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(class_names))
	plt.xticks(tick_marks, class_names, rotation=45)
	plt.yticks(tick_marks, class_names)
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.savefig(filename)

def evaluateVariables(X,y):
	
	X = DataFrame(scale(X), index=origin.index, columns=origin.columns)

	print "all variables : " + str(list(X.columns.values))
	print "\n"

	kBest1 = []
	kBest2 = []
	rfe = []

	for i in reversed (range (2,len(X.columns))):
		kbest_i = SelectKBest(f_classif, k=i)
		kbest_i.fit_transform(X,y)
		mask = kbest_i.get_support()
		#kbest2 = SelectKBest(f_classif, k=2).fit_transform(X, y)
		#print "Select K best with k = "+ str(i) + " using f_classif"
		#print str(list(X.columns[mask].values))
		kBest1.append(list(X.columns[mask].values))
		#print "\n"

	for i in reversed (range (2,len(X.columns))):
		kbest_i = SelectKBest(mutual_info_classif, k=i)
		kbest_i.fit_transform(X,y)
		mask = kbest_i.get_support()
		#kbest2 = SelectKBest(f_classif, k=2).fit_transform(X, y)
		#print "Select K best with k = "+ str(i) + " using mutual_info_classif"
		#print str(list(X.columns[mask].values))
		kBest2.append(list(X.columns[mask].values))
		#print "\n"

	for i in reversed (range (2,len(X.columns))):	
		#print "Recursive Feature Selection"
		#estimator = SVR(kernel="linear")
		axgb = XGBClassifier(nthread=6)
		selector = RFE(axgb, i, step=1)
		selector.fit(X, y)
		mask = selector.get_support()
		#print "selecting " +str(i)+" variables"
		#print X.columns[mask] 
		rfe.append(list(X.columns[mask].values))
		#print "\n"

	return kBest1,kBest2,rfe

def prepare_canvas():
	plt.xlabel('Recall')
	plt.ylabel('Precision')	

def plot_precision_recall(filename,filepath,classifier,Xtest,Ytest,lineColor,flush=False):

	if( gridSearch == True):
		classifier = classifier.best_estimator_

	y_score = classifier.predict_proba(Xtest)
	precision, recall, _ = precision_recall_curve(Ytest, y_score[:, 1])

	# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument

	legend = str(type(classifier))
	legend = legend.split('.')[-1]
	legend = legend.replace('>',"")
	legend = legend.replace('\'',"")
	step_kwargs = ({'step': 'post'}
				if 'step' in signature(plt.fill_between).parameters
				else {})
	plt.step(recall, precision, color=lineColor,
			where='post',label=legend)
	# plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
	
	plt.legend()
	
	if flush:
		plt.ylim([0.0, 1.05])
		plt.xlim([0.0, 1.0])		
		finalPath = os.path.join(filepath,filename)
		plt.savefig(finalPath)
		plt.clf()


def train_and_test(origin,origin_Y,experimentTag=""):


	splits = 10

	if(mode == 'attempts'):
		skf = StratifiedKFold(n_splits=splits,shuffle=True,random_state=seed)		
	
	if(mode == 'ce'):
		skf = StratifiedKFold(n_splits=splits,shuffle=True,random_state=seed)		

	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, early_stopping = True, validation_fraction = 0.22)
	knn = KNeighborsClassifier()
	logistic = linear_model.LogisticRegression()
	xgb = XGBClassifier(nthread=6)


	if( gridSearch == True):
		MLP_params = {
			"solver" : ['sgd', 'lbfgs'],
			"learning_rate" : ['invscaling', 'adaptive'],
			"early_stopping" : [True],
			"hidden_layer_sizes" : [(100,), (100, 100)]
		}

		LR_params = {
			"penalty" : ['l2'],
			"solver" : ['lbfgs', 'liblinear', 'sag']
		}


		KNN_params = {
			"n_neighbors" : [3,5,7],
			"p" : [1,2]
		}

		XGB_params = {
			"n_estimators" : [10, 50, 100],
			"booster" : ['gbtree', 'gblinear']
		}

		clf = GridSearchCV(MLPClassifier(alpha=1e-5,random_state=1, validation_fraction = 0.22),n_jobs=8,cv=skf,param_grid = MLP_params,refit=True,return_train_score=True)
		knn = GridSearchCV(KNeighborsClassifier(),n_jobs=8,cv=skf,param_grid = KNN_params,refit=True,return_train_score=True)
		logistic = GridSearchCV(linear_model.LogisticRegression(),n_jobs=8,cv=skf,param_grid = LR_params,refit=True,return_train_score=True)
		xgb = GridSearchCV(XGBClassifier(),cv=skf,param_grid = XGB_params,refit=True,return_train_score=True)


	step = 1

	mlp_results = Results(splits)
	knn_results = Results(splits)
	logistic_results = Results(splits)
	xgb_results = Results(splits)

	mlp_prediction_list = []
	knn_prediction_list = []
	lrg_prediction_list = []	
	xgb_prediction_list = []

	inputFilePath = os.path.basename(inputFile)
	inputFilePath = inputFilePath.replace(".csv","")
	filename = experimentTag+"_" +inputFilePath + ".svg"

	prepare_canvas()

	for train, test in skf.split(origin, origin_Y):

		if(verbose == True):
			print "===========================step " + str(step)+"============================"

		step+=1
		
		clf.fit(origin.iloc[train], origin_Y.iloc[train])
		mlp_prediction = clf.predict(origin.iloc[test])
		mpl_confusionmatrix = confusion_matrix(origin_Y.iloc[test], mlp_prediction)
		mlp_prediction_list.extend(mlp_prediction)				

		mlp_results.addResult(accuracy_score(origin_Y.iloc[test],mlp_prediction),precision_score(origin_Y.iloc[test],mlp_prediction,average='binary'),
			recall_score(origin_Y.iloc[test],mlp_prediction,average='binary'),cohen_kappa_score(origin_Y.iloc[test],mlp_prediction))

		if(verbose == True):
			print "\n"
			print "Multi-layer Perceptron results:"
			print "accuracy_score: " +  str(mlp_results.accuracies[mlp_results.lastPos-1])
			print "precision_score :" + str(mlp_results.precisions[mlp_results.lastPos-1])
			print "recall_score: "+ str(mlp_results.recalls[mlp_results.lastPos-1])
			print "kappa_score: "+ str(mlp_results.kappas[mlp_results.lastPos-1])
			print "\n"
			
		
		if(precisionRecallPath is not None and step - 1 == splits):
			plot_precision_recall(filename,precisionRecallPath,clf,origin.iloc[test],origin_Y.iloc[test],'r')



		knn.fit(origin.iloc[train], origin_Y.iloc[train])
		knn_prediction = knn.predict(origin.iloc[test])
		knn_confusionmatrix = confusion_matrix(origin_Y.iloc[test], knn_prediction)
		knn_prediction_list.extend(knn_prediction)

		knn_results.addResult(accuracy_score(origin_Y.iloc[test],knn_prediction), precision_score(origin_Y.iloc[test],knn_prediction,average='binary'),
			recall_score(origin_Y.iloc[test],knn_prediction,average='binary'),cohen_kappa_score(origin_Y.iloc[test],knn_prediction))

		if(verbose == True):
			print "\n"
			print "K Nearest neighbor results:"
			print "accuracy_score: " +  str(knn_results.accuracies[knn_results.lastPos-1])
			print "precision_score :" + str(knn_results.precisions[knn_results.lastPos-1])
			print "recall_score: "+ str(knn_results.recalls[knn_results.lastPos-1])
			print "kappa_score: "+ str(knn_results.kappas[knn_results.lastPos-1])
			print "\n"

		if(precisionRecallPath is not None and step - 1 == splits):
			plot_precision_recall(filename,precisionRecallPath,knn,origin.iloc[test],origin_Y.iloc[test],'g')

		logistic.fit(origin.iloc[train], origin_Y.iloc[train])
		logistic_prediction = logistic.predict(origin.iloc[test])
		logistic_confusionmatrix = confusion_matrix(origin_Y.iloc[test], logistic_prediction)
		lrg_prediction_list.extend(logistic_prediction)

		logistic_results.addResult(accuracy_score(origin_Y.iloc[test],logistic_prediction),precision_score(origin_Y.iloc[test],logistic_prediction,average='binary'),
			recall_score(origin_Y.iloc[test],logistic_prediction,average='binary'),cohen_kappa_score(origin_Y.iloc[test],logistic_prediction))

		if(verbose == True):
			print "\n"
			print "Logistic regression results:"
			print "accuracy_score: " +  str(logistic_results.accuracies[logistic_results.lastPos-1])
			print "precision_score :" + str(logistic_results.precisions[logistic_results.lastPos-1])
			print "recall_score: "+ str(logistic_results.recalls[logistic_results.lastPos-1])
			print "kappa_score: "+ str(logistic_results.kappas[logistic_results.lastPos-1])
			print "\n"
		
		if(precisionRecallPath is not None and step - 1 == splits):
			plot_precision_recall(filename,precisionRecallPath,logistic,origin.iloc[test],origin_Y.iloc[test],'b')

		xgb.fit(origin.iloc[train], origin_Y.iloc[train])
		xgb_prediction = xgb.predict(origin.iloc[test])
		xgb_confusionmatrix = confusion_matrix(origin_Y.iloc[test], xgb_prediction)
		xgb_prediction_list.extend(xgb_prediction)

		xgb_results.addResult(accuracy_score(origin_Y.iloc[test],xgb_prediction),precision_score(origin_Y.iloc[test],xgb_prediction,average='binary'),
			recall_score(origin_Y.iloc[test],xgb_prediction,average='binary'),cohen_kappa_score(origin_Y.iloc[test],xgb_prediction))

		if(verbose == True):
			print "\n"
			print "XGboost results:"
			print "accuracy_score: " +  str(xgb_results.accuracies[xgb_results.lastPos-1])
			print "precision_score :" + str(xgb_results.precisions[xgb_results.lastPos-1])
			print "recall_score: "+ str(xgb_results.recalls[xgb_results.lastPos-1])
			print "kappa_score: "+ str(xgb_results.kappas[xgb_results.lastPos-1])			
			print "\n"
		
		if(precisionRecallPath is not None and step - 1 == splits):						
			plot_precision_recall(filename,precisionRecallPath,xgb,origin.iloc[test],origin_Y.iloc[test],'black',True)
			

			

	print "MLP confusion matrix\n"
	print mpl_confusionmatrix
	print "\n"

	mlp_results.printStatistics()
	print "\n"
	print "\n"


	print "KNN confusion matrix\n"
	print knn_confusionmatrix
	print "\n"
	knn_results.printStatistics()
	print "\n"
	print "\n"


	print "Logistic Regression confusion matrix\n"
	print logistic_confusionmatrix
	print "\n"
	logistic_results.printStatistics()	
	print "\n"
	print "\n"

	print "XGBoost\n"
	print xgb_confusionmatrix
	print "\n"
	xgb_results.printStatistics()
	print "\n"
	
	print "XGBoost Feature importances: "
	
	headers = ['Score','Parameter']
	importanceGain = None
	if(gridSearch == True):
		importanceGain = xgb.best_estimator_.get_booster().get_score(importance_type='gain')
	else:
		importanceGain = xgb.get_booster().get_score(importance_type='gain')
	data = sorted([(v,k) for k,v in importanceGain.items()], reverse=True) # flip the code and name and sort
	print "Feature Importance (gain):\n"
	print tabulate.tabulate(data, headers=headers)	 
	print "\n"

	weightGain = None
	if(gridSearch == True):
		weightGain = xgb.best_estimator_.get_booster().get_score(importance_type='weight')
	else:
		weightGain = xgb.get_booster().get_score(importance_type='weight')
	data = sorted([(v,k) for k,v in weightGain.items()], reverse=True) # flip the code and name and sort
	print "Feature Importance (weight):\n"
	print tabulate.tabulate(data, headers=headers)	 
	print "\n"

	coverGain = None

	if(gridSearch == True):
		coverGain = xgb.best_estimator_.get_booster().get_score(importance_type='cover')
	else:
		coverGain = xgb.get_booster().get_score(importance_type='cover')
	data = sorted([(v,k) for k,v in coverGain.items()], reverse=True) # flip the code and name and sort
	print "Feature Importance (cover):\n"
	print tabulate.tabulate(data, headers=headers)	 
	print "\n"

	output_table = DataFrame({'mlp':mlp_prediction_list,'knn':knn_prediction_list,'logistic_regression':lrg_prediction_list , 'xgb':xgb_prediction_list})
	return output_table


def run_experiment(origin_copy, outputDataFrameName , outputFoldsFileNameSufix ):
	
	print "STARTING TRAINING"
	print "considering variables: " + str(list(origin_copy.columns.values))

	origin_copy = DataFrame(scale(origin_copy), index=origin_copy.index, columns=origin_copy.columns)	
	output_frame = train_and_test(origin_copy,origin_Y,outputDataFrameName)
	
	if outputFolds:
		inputFilePath = os.path.basename(inputFile)	
		inputFilePath = inputFilePath.replace(".csv","")	
		filename = "FOLDS_"+gridSearchStr + inputFilePath + outputFoldsFileNameSufix +".csv"
		filename = os.path.join(outputFolds,filename)
		output_frame = concat([frameNamesClass,output_frame],axis=1)				
		output_frame.to_csv(filename,sep=";",index=False,header="k1_names,k2_names,classification,mlp,knn,regression,xgb")


frameNamesClass = origin[['k1_name','k2_name','classification']]
# namesAndClassification = concat(frameNamesClass)

if mode == "ce":
	classes = [0,1]
if mode == 'attempts':	
	print "number of tries: " 
	print origin["attempts_of_5"].value_counts()
	origin = origin.drop(["attempts_of_5"],axis=1)
	classes = [0,1]


origin = origin.drop(["k1_name"],axis = 1)
origin = origin.drop(["k2_name"],axis = 1)


print "classes count: " 
print origin["classification"].value_counts()
origin_Y = origin.classification



origin = origin.drop(["classification"], axis =1 )

kBest1, kBest2, rfe = evaluateVariables(origin,origin_Y)

if outputFolds:
	output_frame = DataFrame()

print "testing for Recursive Feature Selection ================================================================"
for variables in rfe:

	#origin_copy = origin.drop( variables ,axis=1)
	origin_copy = origin[variables]

	if (len(list(origin_copy.columns.values)) == features):		
		run_experiment(origin_copy, gridSearchStr+"RFE","_RFE")

if(len(userVariables) > 0):
	print "testing for user selected variables ==============================================================================="
	

	dropVars = list(set(origin.columns.values) - set(userVariables))
	
	origin_copy = origin
	
	for drop in dropVars:
		origin_copy = origin_copy.drop([drop],axis = 1)
	
	run_experiment(origin_copy, gridSearchStr+"USER_VARS", "_USER_VARS_NO_RFE")
	

print "testing for all variables ==============================================================================="

run_experiment(origin, gridSearchStr+"ALL_VARS", "_ALL_VARS_NO_RFE")