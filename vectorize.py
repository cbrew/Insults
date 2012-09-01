"""
Code for the insults competition run by Kaggle in August 2012.

So far, winning combination is grid search + sgd regressor.


Best run appears to be this one, with 2750 iterations. These parameters are chosen for
all folds. I like the stability

2012-08-18 12:57:42,500 : INFO : fold 0
2012-08-18 13:11:59,331 : INFO : 	0 clf__alpha: 1.2589254117941662e-07  = 10**6.9
2012-08-18 13:11:59,332 : INFO : 	0 clf__n_iter: 2750
2012-08-18 13:11:59,332 : INFO : 	0 clf__penalty: 'l1'
2012-08-18 13:11:59,332 : INFO : 	0 vect__analyzer: 'char'
2012-08-18 13:11:59,332 : INFO : 	0 vect__lowercase: False
2012-08-18 13:11:59,332 : INFO : 	0 vect__max_n: 5

Uses the default squared loss, eta and learning rate. Should really try huber...



Performance by fold:

2012-08-18 13:12:14,983 : INFO : 0 test 0.921842
2012-08-18 13:26:11,407 : INFO : 1 test 0.889459
2012-08-18 13:39:16,335 : INFO : 2 test 0.888655
2012-08-18 13:52:58,508 : INFO : 3 test 0.900914
2012-08-18 14:06:45,541 : INFO : 4 test 0.887678

Expected.

2012-08-18 14:06:45,541 : INFO : Expected score 0.897710

Ideas
-----

- use character n-grams because they are robust and simple.
- do self-training using current best estimate
- use SGD regression, and tune the bejazus out of it.
- use custom classifiers based on sklearn raw ones.

Questions
=========
- why not a classifier? Trying it.
- what is the best thing to do about the self-training?
- did we really self-train the first time? Maybe we did, but it doesn't work
- will it help to cross-validate in order to produce self-train predictions? Apparently not so far

Desired interface
=================

Read set of training files.
Read grid configuration.


"""
import pandas
from sklearn import feature_extraction,linear_model,cross_validation,ensemble,svm,pipeline,grid_search
import ml_metrics
import numpy as np
import os
import itertools
import logging
import datetime
import random
import joblib
from sklearn.feature_selection import chi2,SelectKBest


self_train = False

logging.basicConfig(filename="vectorize.log",mode='w',format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
train = pandas.read_table('Data/train.csv',sep=',')
best = pandas.read_table('best/best.csv',sep=',')
leaderboard = pandas.read_table('Data/test.csv',sep=',')


def round_predictions(ypred):
	"""
	normalize range of predictions to 0-1
	"""
	yrange = ypred.max() - ypred.min()
	ypred /= yrange
	ypred -= ypred.min()
	# protection against rounding, ENSURE nothing out of range.
	ypred[ypred > 1] = 1
	ypred[ypred < 0] = 0
	return ypred
	
	
class MySGDRegressor(linear_model.SGDRegressor):
	"""
	Ensure that predictions are suitably rounded. Doing it here makes sure
	that the quantity that is optimized is also the one that is wanted.
	"""
	def predict(self,X):
		return round_predictions(linear_model.SGDRegressor.predict(self,X))


class MySGDClassifier(linear_model.SGDClassifier):
	"""
	According to the docs, this has probability of positive class 
	as the return value for predict-proba.
	"""
	def predict(self,X):
			return linear_model.SGDClassifier.predict_proba(self,X)

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
    'vect__ngram_range': ((1,1),(1,2),(1,3),(2,3)),  # words,bigrams,trigrams
    #    'tfidf__use_idf': (True, False),
    #    'tfidf__norm': ('l1', 'l2'),
    # 'clf__C': (1e-2,1,1e+2),
    # 'clf__C': (10,15,20,21,22,23,24,25,26,27,28,29,30,35,40,45,50,55,60,65,70,75,80,85,100,1000),
    # 'clf__tol': (1e-3,1e-4),
    'clf__penalty': ("l1","elasticnet"),
    'clf__alpha': 10.0**-np.arange(7.0,8.3,step=0.2),
    'clf__n_iter': (1000,1250,1500,1750,2000),
}




# Grid for SGD using character n-grams

grid["sgd_c"] = {
    'vect__analyzer': ['char'],
    'vect__lowercase': [False],
    'vect__ngram_range': ((3,5),),  # words,bigrams,trigrams, etc.
	'clf__loss': ("modified_huber","log"),
    'clf__penalty': ("l1","l2"),
    'clf__alpha': 10.0**-np.linspace(6.,8.,num=10),
    'clf__n_iter': (100,200,400),
}



def save_predictions(ypred,i):
    for x in itertools.count(1):
        filename = "%s_fold_%d_%d"   % (clf_name,i,x)
        if os.path.exists(filename):
            next
        else:			
            np.save(filename,ypred)
            break


fixed = {}
fixed['sgd_c'] = {"shuffle": True}
fixed['sgd_w'] = {}
clfs = dict(sgd=(MySGDRegressor,fixed['sgd_w'],grid["sgd_w"]),
            sgd_char=(MySGDClassifier,fixed['sgd_c'],grid["sgd_c"]))

# v = pandas.read_table('embeddings-scaled.EMBEDDING_SIZE=50.txt',sep=' ',header=None,index_col='X.1')


clfc,clfix,clfp = clfs["sgd_char"]


# print clfc,clfix,clfp





pipeline1 = pipeline.Pipeline([
		    		('vect', feature_extraction.text.CountVectorizer(
		    				lowercase=False,
		    				analyzer='char',
		    				ngram_range=(1,5),
		    				)
		    		),
		    		('tfidf', feature_extraction.text.TfidfTransformer(sublinear_tf=True,norm="l2")),
		    		('clf',linear_model.RidgeCV(score_func=ml_metrics.auc)),
			])

clf =  grid_search.GridSearchCV(pipeline1, clfp, n_jobs=2,score_func=ml_metrics.auc,verbose=3)  


clf = pipeline1

def training():
			
	ss = 0
	n = 0
	kf = cross_validation.KFold(len(train),5,indices=False)
	for i,(train_i,test_i) in enumerate(kf):
		ftrain = train[train_i]
		logging.info('fold %d' % i)
		clf.fit(ftrain.Comment,ftrain.Insult)
		# print i,clf.steps[2][1].cv_values_
		# best_parameters = clf.best_estimator_.get_params()
		# for param_name in sorted(clfp.keys()):
		#	logging.info("\t%d %s: %r" % (i,param_name, best_parameters[param_name]))
		ypred = clf.predict(ftrain.Comment) 
		logging.info("%d train=%f" % (i, ml_metrics.auc(np.array(ftrain.Insult),ypred)))
		ypred = clf.predict(train[test_i].Comment)
		est = ml_metrics.auc(np.array(train[test_i].Insult),ypred)
		logging.info("%d test %f" % (i,est))
		ss += est
		n  += 1
	logging.info('Expected score %f' % (ss/n))
	


def predict():
	logging.info("Starting leaderboard")
	
	clf.fit(train.Comment,train.Insult)
	ypred = clf.predict(leaderboard.Comment)
	ypred = round_predictions(ypred)
	# we create a submission...
	submission = pandas.read_table('Data/sample_submission_null.csv',sep=',')
	submission['Insult'] = ypred
	for x in itertools.count(1):
			filename = "submissions/submission%d.csv" % x
			if os.path.exists(filename):
				next
			else:			
				submission.to_csv(filename,index=False)
				logging.info("result saved to %s" % filename)
				model_filename = "models/model%d" % x
				# XXX following line does not save the right model when self-training.
				# Instead, it saves the model for fold k-1 of the cross-val.
				# work is needed to create a model based on all folds.
				# joblib.dump(clf.best_estimator_,model_filename,compress=3)
				# logging.info("model saved to %s" % model_filename)
				break

