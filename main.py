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

# 
nfeats = 5000

"""
Current best uses GradientBoostingClassifier and 5000 features selected as most frequent.
{'n_estimators': 400, 'subsample': 0.5, 'random_state': 0, 'learn_rate': 0.025}.
Did not record which features went in (foolish, right?), but there were obviously a lot of them...

2012-08-12 18:03:46,390 : INFO : started fold 0 657.227415
2012-08-12 18:03:46,772 : INFO : 0 train auc 0.936763
2012-08-12 18:03:57,129 : INFO : 0 test auc 0.894781
2012-08-12 18:04:32,818 : INFO : limiting 155562 features to 5000
2012-08-12 18:14:55,571 : INFO : started fold 1 1326.408069
2012-08-12 18:14:55,971 : INFO : 1 train auc 0.939800
2012-08-12 18:15:07,957 : INFO : 1 test auc 0.877702
2012-08-12 18:15:45,067 : INFO : limiting 154198 features to 5000
2012-08-12 18:26:01,762 : INFO : started fold 2 1992.599039
2012-08-12 18:26:02,154 : INFO : 2 train auc 0.940173
2012-08-12 18:26:12,892 : INFO : 2 test auc 0.866636
2012-08-12 18:26:49,283 : INFO : limiting 157722 features to 5000
2012-08-12 18:37:09,402 : INFO : started fold 3 2660.238973
2012-08-12 18:37:09,795 : INFO : 3 train auc 0.937719
2012-08-12 18:37:20,312 : INFO : 3 test auc 0.877904
2012-08-12 18:37:57,923 : INFO : limiting 154853 features to 5000
2012-08-12 18:48:13,541 : INFO : started fold 4 3324.377875
2012-08-12 18:48:13,959 : INFO : 4 train auc 0.940377
2012-08-12 18:48:25,224 : INFO : 4 test auc 0.852312
"""


train = pandas.read_table('Data/train.csv',sep=',')
# clf_class = linear_model.LogisticRegression
# params = dict(penalty='l1')
clf_class = ensemble.GradientBoostingClassifier
params = dict(n_estimators=400,learn_rate=0.025,subsample=0.5,random_state=0)


def do_train(fold,train_i,test_i):
		rf = clf_class(**params)
		logging.info('classifier is %r' % rf)
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
		
		
start = time.time()

kf = cross_validation.KFold(len(train),5,indices=False)
print nfeats,params
# pl.figure()		
for i,(train_i,test_i) in enumerate(kf):
		logging.info("fold %d started" % i)	
		do_train(i,train_i,test_i)
		# pl.plot((np.arange(test_deviance.shape[0]) + 1)[::5], test_deviance[::5], '-', label=("fold %d" % fold))
		logging.info("fold %d done" % i)
		

		
#pl.legend(loc='lower right')
# pl.xlabel('Boosting Iterations (learn rate=%f)' % params["learn_rate"])
#pl.ylabel('Test Set AUC')
#pl.show()

		
	
	
test = pandas.read_table('Data/test.csv',sep=',')
rf = clf_class(**params)
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
