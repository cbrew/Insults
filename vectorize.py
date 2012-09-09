"""
Code for the insults competition run by Kaggle in August 2012.


Ideas
-----

- use character n-grams because they are robust and simple.
- use multiple good classifiers and average them.
 





"""

import pandas
from sklearn import feature_extraction,linear_model,cross_validation,pipeline
import ml_metrics
import numpy as np
import os
import itertools
import logging
from sklearn.svm import LinearSVC,SVR,l1_min_c
import pylab as pl
from collections import Counter
from IPython.core.display import display




def initialize():
	"""
	Set everything up.
	"""
	logging.basicConfig(filename="vectorize.log",mode='w',format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	train = pandas.read_table('Data/train.csv',sep=',')
	leaderboard = pandas.read_table('Data/test.csv',sep=',')
	return train,leaderboard

def scale_predictions(ypred):
	"""
	normalize range of predictions to 0-1
	"""

	yrange = ypred.max() - ypred.min()
	ypred -= ypred.min()
	ypred /= yrange
	
	# protection against rounding, ENSURE nothing out of range.
	ypred[ypred > 1] = 1
	ypred[ypred < 0] = 0
	return ypred
	
	
class MySGDRegressor(linear_model.SGDRegressor):
	"""
	An SGD regressor that 
	a) transforms the output into 0..1
	b) fits in stages so you can see the effect of number of iterations.
	"""
	def __init__(self,n_iter_per_step=50,max_iter=500,alpha=0.001,penalty='l2',**kwargs):
		self.max_iter=max_iter
		self.kwargs = kwargs
		self.n_iter_per_step = n_iter_per_step
		self.alpha = alpha
		self.penalty = penalty
		self.reset_args()
	def reset_args(self):
		# enforce condition that n_iter_per_step must be a factor of n_iter
		assert self.max_iter % self.n_iter_per_step == 0
		linear_model.SGDRegressor.__init__(self,
											alpha=self.alpha,
											penalty=self.penalty,
											n_iter=self.n_iter_per_step,
											**self.kwargs)
	def fit(self,X,y):
		self.coef_ = None
		self.intercept_ = None
		self.stages_ = []
		for i in range(0,self.max_iter,self.n_iter):
			
			if self.coef_ != None:
				assert(self.intercept_ != None)
				linear_model.SGDRegressor.fit(self,X,y,coef_init=self.coef_,intercept_init=self.intercept_)
			else:
				linear_model.SGDRegressor.fit(self,X,y)
			# record coefs and intercept for later
			self.stages_.append((i+self.n_iter,self.coef_.copy(),self.intercept_.copy()))
			logging.info('done %d/%d steps' % (i+self.n_iter,self.max_iter))
			logging.info('training set auc %f' % self.auc(X,y))
	def auc(self,X,y):
		yhat = self.predict(X)
		return ml_metrics.auc(np.array(y),yhat)

	def staged_predict(self,X):
		"""
		Predict after each of the stages.
		"""
		return [(n_iter_,self.predict(X,coef=coef_,intercept=intercept_)) for (n_iter_,coef_,intercept_) in self.stages_]

	def staged_auc(self,X,y):
		"""
		calculate the AUC after each of the stages.
		"""
		y = np.array(y)
		return [ (n, ml_metrics.auc(y,p)) for n,p in self.staged_predict(X)]

		
	def predict(self,X,coef=None,intercept=None):
		"""
		a) do the prediction based on given coefs and intercept, if provided.
		b) Scale the predictions so that they are in 0..1. 

		"""
		if coef != None:
			assert intercept != None
			self.intercept_ = intercept
			self.coef_ = coef

		return scale_predictions(linear_model.SGDRegressor.predict(self,X))






def save_predictions(ypred,i):
    for x in itertools.count(1):
        filename = "%s_fold_%d_%d"   % (clf_name,i,x)
        if os.path.exists(filename):
            next
        else:			
            np.save(filename,ypred)
            break










class MyPipeline(pipeline.Pipeline):
	def staged_auc(self,X,y):
		"""
		"""
		Xt = X
		for name, transform in self.steps[:-1]:
			Xt = transform.transform(Xt)
		return self.steps[-1][-1].staged_auc(Xt,y)


clf = MyPipeline([
		    		('vect', feature_extraction.text.CountVectorizer(
		    				lowercase=False,
		    				analyzer='char',
		    				ngram_range=(1,5),
		    				)
		    		),
		    		('tfidf', feature_extraction.text.TfidfTransformer(sublinear_tf=True,norm="l2")),
					("clf",MySGDRegressor(
		    				alpha=5e-8,
		    				penalty='l2',
		    				max_iter=1500,
		    				n_iter_per_step=50)),

					])






def training():
	"""
	The model is refitted in the predict stage. The purpose of this code is to choose
	a nice set of parameters.
	"""
	

	best_iters = []
	best_iter = clf.steps[-1][-1].max_iter / 	clf.steps[-1][-1].n_iter_per_step	
	kf = cross_validation.KFold(len(train.Insult),5,indices=False)
	for i,(train_i,test_i) in enumerate(kf):
		ftrain = train[train_i]
		logging.info('fold %d' % i)
		clf.fit(ftrain.Comment,ftrain.Insult)
		ypred = clf.predict(ftrain.Comment) 
		logging.info("%d train auc=%f" % (i, ml_metrics.auc(np.array(ftrain.Insult),ypred)))
		ypred = clf.predict(train[test_i].Comment)

		# the policy is that we are going to select the number of iterations for the
		# predict stage by examining the curve for the training folds.

		xs,ys = zip(*clf.staged_auc(train[test_i].Comment,train[test_i].Insult))
		xs = np.array(xs)
		ys = np.array(ys)
		
		best_this_fold = ys.argmax()
		best_iter = min(best_this_fold,best_iter)
		best_iters.append(xs[best_this_fold])
		

		display(clf.steps[-1][-1])
		pl.title("Fold %d chosen %d %0.5f this fold: %d %0.5f" % (
																	i,
																	xs[best_iter],ys[best_iter],
																	xs[best_this_fold],ys[best_this_fold],
																))
																	
		pl.plot(xs,ys)
		xs,ys = zip(*clf.staged_auc(ftrain.Comment,ftrain.Insult))
		pl.plot(xs,ys)
		pl.xlabel('Number of iterations')
		pl.ylabel('AUC')
		pl.grid(True)
		pl.show()
	
	



	# tell the inner classifier how many steps would be best...
	clf.steps[-1][-1].max_iter = xs[best_iter]
	clf.steps[-1][-1].reset_args()

	# work out what the score we expect is...
	ss = 0
	n = 0
	for i,(train_i,test_i) in enumerate(kf):
		ftrain = train[train_i]
		logging.info('fold %d' % i)
		clf.fit(ftrain.Comment,ftrain.Insult)
		ypred = clf.predict(train[test_i].Comment)
		est = ml_metrics.auc(np.array(train[test_i].Insult),ypred)
		logging.info("%d test auc %d iter %f" % (i,clf.steps[-1][-1].max_iter,est))
		ss += est
		n  += 1


	# optimal choice of iteration number not guaranteed, because it is a
	# mimimum over folds. Minimizing seems a sensible thing, because
	# overfitting is our expected problem, but not tested...


	logging.info('Expected auc %f max_iter %d max_iters %r' % (ss/n,xs[best_iter],best_iters))

	


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


