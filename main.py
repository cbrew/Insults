import pandas
from sklearn import ensemble,cross_validation,linear_model,feature_selection,svm
import features
import ml_metrics
import numpy as np
import time
import os
import itertools
import logging
import sys
import pylab as pl
import concurrent.futures as futures

params = dict(n_estimators=500,learn_rate=0.01,subsample=0.5,random_state=0)
nfeats = 2000

train = pandas.read_table('Data/train.csv',sep=',')

def do_train(fold,train_i,test_i):
		
		rf = ensemble.GradientBoostingClassifier(**params)
		ftrain = train[train_i]
		ftest = train[test_i]
		ytrain = np.array(ftrain.Insult).astype('float64')
		bag = features.train_bag(ftrain.Comment,nfeats,y=ytrain)	
		fea =  features.bag_representation(bag, ftrain.Comment)
		rf.fit(fea,ytrain)
		logging.info("started fold %d %f" % (fold,time.time() - start))
		ypred = rf.predict_proba(fea)[:,1]
		logging.info("%d train auc %f" % (fold,ml_metrics.auc(ytrain,ypred)))
		
		y_test = np.array(ftest.Insult).astype('float64')
		X_test =  features.bag_representation(bag, ftest.Comment)
		ypred = rf.predict_proba(X_test)[:,1]
		logging.info("%d test auc %f" % (fold,ml_metrics.auc(y_test,ypred)))
		df = pandas.DataFrame(dict(importance=rf.feature_importances_,name=X_test.columns))
		df = df.sort("importance")
		logging.info("importances\n%r" % (df.tail(100)))
		
		# compute test set auc for each fold
		test_deviance = np.zeros((params['n_estimators'],), dtype=np.float64)
		for i, y_pred in enumerate(rf.staged_decision_function(X_test)):
			test_deviance[i] = ml_metrics.auc(y_test, y_pred)
		return test_deviance,fold

start = time.time()

kf = cross_validation.KFold(len(train),5,indices=False)
print nfeats,params,start
pl.figure()		
for i,(train_i,test_i) in enumerate(kf):
		logging.info("fold %d started" % i)	
		test_deviance,fold = do_train(i,train_i,test_i)
		pl.plot((np.arange(test_deviance.shape[0]) + 1)[::5], test_deviance[::5], '-', label=("fold %d" % fold))
		logging.info("fold %d done" % i)
	


		
pl.legend(loc='lower right')
pl.xlabel('Boosting Iterations (learn rate=%f)' % params["learn_rate"])
pl.ylabel('Test Set AUC')
pl.show()

		
	
	
test = pandas.read_table('Data/test.csv',sep=',')
rf = ensemble.GradientBoostingClassifier(**params)
ytrain = np.array(train.Insult).astype('float64')
bag = features.train_bag(train.Comment,nfeats,y=ytrain)
fea = features.bag_representation(bag, train.Comment)

rf.fit(fea,ytrain)
fea = features.bag_representation(bag, test.Comment)
ypred = rf.predict_proba(fea)[:,1]
submission = pandas.read_table('Data/sample_submission_null.csv',sep=',')
submission['Insult'] = ypred

for x in itertools.count(1):
		filename = "submissions/submission%d.csv" % x
		if os.path.exists(filename):
			next
		else:
			logging.info("saved to %s" % filename)
			submission.to_csv(filename,index=False)
			break
print "Done"
