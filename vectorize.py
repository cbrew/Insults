"""
Code for the insults competition run by Kaggle in August 2012.


Ideas
-----

- use character n-grams because they are robust and simple.
- use multiple good classifiers and average them.




Desired interface
=================

Read set of training files.
Read grid configuration.


"""

import pandas
from sklearn import feature_extraction,linear_model,cross_validation,pipeline
import ml_metrics
import numpy as np
import os
import itertools
import logging




def initialize():
	"""
	Set everything up.
	"""
	logging.basicConfig(filename="vectorize.log",mode='w',format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	train = pandas.read_table('Data/train.csv',sep=',')
	leaderboard = pandas.read_table('Data/test.csv',sep=',')
	return train,leaderboard

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






def save_predictions(ypred,i):
    for x in itertools.count(1):
        filename = "%s_fold_%d_%d"   % (clf_name,i,x)
        if os.path.exists(filename):
            next
        else:			
            np.save(filename,ypred)
            break


class MyCombinedClassifier:
	def __init__(self):
		self.ridge = linear_model.RidgeCV(score_func=ml_metrics.auc,alphas=[1.5,1.1,0.99,0.75])
		self.sgd = linear_model.SGDRegressor(
		    				alpha=3e-7,
		    				penalty='l2', 	# 'l1' is nearly the same as 'l2' for this data.
		    								# but 'l2' has a slighly higher expected score.
		    				n_iter=3000) # 3000
		self.sgd2 = linear_model.SGDRegressor(
		    				alpha=3e-7,
		    				penalty='l1', 	# 'l1' is nearly the same as 'l2' for this data.
		    								# but 'l2' has a slighly higher expected score.
		    				n_iter=3000) # 3000
	def fit(self,X,y):
		self.ridge.fit(X,y)
		self.sgd.fit(X,y)
		self.sgd2.fit(X,y)
		return self
	def predict(self,X):
		y1 = self.ridge.predict(X)
		y2 = self.sgd.predict(X)
		y3 = self.sgd2.predict(X)
		return round_predictions((y1 + y2 +y3)/3)



clf = pipeline.Pipeline([
		    		('vect', feature_extraction.text.CountVectorizer(
		    				lowercase=False,
		    				analyzer='char',
		    				ngram_range=(1,5),
		    				)
		    		),
		    		('tfidf', feature_extraction.text.TfidfTransformer(sublinear_tf=True,norm="l2")),
		    		("clf",MyCombinedClassifier()),
		    		# ("clf",svm.SVR(C=1e5,kernel="linear")),
		    		# ('clf',linear_model.RidgeCV(score_func=ml_metrics.auc,alphas=[1.5,1.1,0.99,0.75])),
			])





def training():
	"""
	The model is refitted in the predict stage. The purpose of this code is to choose
	a nice set of parameters.
	"""
			
	ss = 0
	n = 0
	kf = cross_validation.KFold(len(train.Insult),5,indices=False)
	for i,(train_i,test_i) in enumerate(kf):
		ftrain = train[train_i]
		logging.info('fold %d' % i)
		clf.fit(ftrain.Comment,ftrain.Insult)
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

	# we create a submission...
	submission = pandas.read_table('Data/sample_submission_null.csv',sep=',')
	submission['Insult'] = ypred
	for x in itertools.count(1):
			filename = "submissions/submission%d.csv" % x
			if os.path.exists(filename):
				next
			else:			
				submission.to_csv(filename,index=False)
				logging.info('Saved %s' % filename)
				break

if __name__ == "__main__":
	train,leaderboard = initialize()
	training()
	predict()


