"""
Code for the insults competition run by Kaggle in August 2012.

Winning combination is grid search + sgd regressor.

"""
import pandas
from sklearn import feature_extraction,linear_model,cross_validation,ensemble,svm,pipeline,grid_search
import ml_metrics
import numpy as np
import os
import itertools
import logging
import datetime
import pickle
import random
import joblib

self_train = False

logging.basicConfig(filename="vectorize.log",mode='w',format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

train = pandas.read_table('Data/train.csv',sep=',')

best = pandas.read_table('best/best.csv',sep=',')

if self_train:
		# when we self_train, crucial to shuffle before cross-validation
		logging.info('original training %r',train.shape)
		train = pandas.concat([train,best],ignore_index=True)
		train['Temp'] = random.shuffle(range(train.shape[0]))
		train = train.sort('Temp')
		del train['Temp']
		logging.info('new training %r',train.shape)

leaderboard = pandas.read_table('Data/test.csv',sep=',')

class MyLogisticRegression(linear_model.LogisticRegression):
    def predict(self, X):
        return linear_model.LogisticRegression.predict_proba(self, X)[:,1]


clf_name = "sgd_c"

grid = {}

grid["sgd_w"] = {
    # uncommenting more parameters will give better exploring power but will
    # increase processing time in a combinatorial way
    # 'vect__max_df': (0.5, 0.75, 1.0),
    # 'vect__analyzer':('word'),
    #  'vect__lowercase': (True,False),
	
    # 'vect__max_features': (None, 10, 50, 100, 500,1000,5000, 10000, 50000),
    'vect__max_n': (1,2,3),  # words,bigrams,trigrams
    #    'tfidf__use_idf': (True, False),
    #    'tfidf__norm': ('l1', 'l2'),
    # 'clf__C': (1e-2,1,1e+2),
    # 'clf__C': (10,15,20,21,22,23,24,25,26,27,28,29,30,35,40,45,50,55,60,65,70,75,80,85,100,1000),
    # 'clf__tol': (1e-3,1e-4),
    'clf__penalty': ("l1","elasticnet"),
    'clf__alpha': 10.0**-np.arange(7.0,8.3,step=0.2),
    'clf__n_iter': (1000,1250,1500,1750,2000),
}



grid["sgd_c"] = {
    # uncommenting more parameters will give better exploring power but will
    # increase processing time in a combinatorial way
    # 'vect__max_df': (0.5, 0.75, 1.0),
    'vect__analyzer': ['char'],
    'vect__lowercase': [False],
    # 'vect__max_features': (None, 10, 50, 100, 500,1000,5000, 10000, 50000),
    'vect__max_n': (5, ),  # words,bigrams,trigrams
    #    'tfidf__use_idf': (True, False),
    #    'tfidf__norm': ('l1', 'l2'),
    # 'clf__C': (1e-2,1,1e+2),
    # 'clf__C': (10,15,20,21,22,23,24,25,26,27,28,29,30,35,40,45,50,55,60,65,70,75,80,85,100,1000),
    # 'clf__tol': (1e-3,1e-4),
    'clf__penalty': ("l1", ),
    'clf__alpha': 10.0**-np.arange(6.8,7.3,step=0.1),
		# 'clf__alpha': 10.0**-np.arange(6.5,8.5,step=1.0),
    'clf__n_iter': (2600,2750,2900),
}


fixed = {}
fixed['sgd_c'] = {"shuffle": True}
fixed['sgd_w'] = {}



clfs = dict(sgd=(linear_model.SGDRegressor,fixed['sgd_w'],grid["sgd_w"]),
            sgd_char=(linear_model.SGDRegressor,fixed['sgd_c'],grid["sgd_c"]))


clfc,clfix,clfp = clfs["sgd_char"]


print clfc,clfix,clfp

pipeline1 = pipeline.Pipeline([
    ('vect', feature_extraction.text.CountVectorizer(lowercase=True,max_n=2)),
    ('tfidf', feature_extraction.text.TfidfTransformer()),
    ('clf',clfc(**clfix)),
])



def save_predictions(ypred,i):
    for x in itertools.count(1):
        filename = "%s_fold_%d_%d"   % (clf_name,i,x)
        if os.path.exists(filename):
            next
        else:			
            np.save(filename,ypred)
            break

clf =  grid_search.GridSearchCV(pipeline1, clfp, n_jobs=2,score_func=ml_metrics.auc,verbose=3)  

			
ss = 0
n = 0
kf = cross_validation.KFold(len(train),5,indices=False)
for i,(train_i,test_i) in enumerate(kf):
	logging.info('fold %d' % i)
	clf.fit(train[train_i].Comment,train[train_i].Insult)
	best_parameters = clf.best_estimator_.get_params()
	for param_name in sorted(clfp.keys()):
		logging.info("\t%d %s: %r" % (i,param_name, best_parameters[param_name]))
	ypred = clf.predict(train[train_i].Comment) 
	logging.info("%d train=%f" % (i, ml_metrics.auc(np.array(train[train_i].Insult),ypred)))
	ypred = clf.predict(train[test_i].Comment)
	est = ml_metrics.auc(np.array(train[test_i].Insult),ypred)
	logging.info("%d test %f" % (i,est))
	ss += est
	n  += 1
	
	
logging.info('Expected score %f' % (ss/n))
logging.info("Starting leaderboard")
clf.fit(train.Comment,train.Insult)
best_parameters = clf.best_estimator_.get_params()
for param_name in sorted(clfp.keys()):
	logging.info("\tL %s: %r" % (param_name, best_parameters[param_name]))
ypred = clf.predict(leaderboard.Comment)
submission = pandas.read_table('Data/sample_submission_null.csv',sep=',')
ypred[ypred > 1] = 1
ypred[ypred < 0] = 0
submission['Insult'] = ypred


# we create a submission...
for x in itertools.count(1):
		filename = "submissions/submission%d.csv" % x
		if os.path.exists(filename):
			next
		else:			
			submission.to_csv(filename,index=False)
			logging.info("result saved to %s" % filename)
			model_filename = "models/model%d" % x
			joblib.dump(clf.best_estimator_,model_filename,compress=3)
			logging.info("model saved to %s" % model_filename)
			
			break
print "Done"
