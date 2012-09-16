"""
Code for the insults competition run by Kaggle for Impermium in August-September 2012.

Prerequisites
-------------

Python 2.7.3 + scikit-learn 0.13-git + ml_metrics 0.1.1 + pandas 0.8.1 + concurrent.futures

 (+ matplotlib 1.2.x for plotting, but not needed to predict)
(all have licenses that permit commercial use)
I ran on a four year old iMac with 4Gb of memory and dual core Intel processor.

Usage
-----
python insults.py  (does training on train.csv, testing on test.csv, output a numbered file in ./Submissions/submissionXX.csv
python insults.py --train Data/fulltrain.csv --predictions cbrew.csv  (trains on fulltrain.csv, tests on test.csv, results
in cbrew.csv. These results are much too good, because the training and test sets overlap )


requires the datafiles in ./Data

writes information into Folds which is used to train the final model.
generates an extensive log file in (by default) 'insults.log'

Ideas
-----

- use character n-grams because they are robust and simple.
- tune SGD carefully.

Issues
------
clumsy to overwrite the fold data each time we tune. 

"""

import pandas
from sklearn import feature_extraction,linear_model,cross_validation,pipeline
import ml_metrics
import numpy as np
import os
import itertools
import logging
import pylab as pl
import argparse
import sys
import score
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
import concurrent.futures




def make_full_training():
	df1 = pandas.read_table('Data/train.csv',sep=',')
	df2 = pandas.read_table('Data/test_with_solutions.csv',sep=',')
	df = pandas.concat([df1,df2])
	df.to_csv('Data/fulltrain.csv',index=False)




def make_clf(args):
	return MyPipeline([
		    		('vect', feature_extraction.text.CountVectorizer(
		    				lowercase=False,
		    				analyzer='char',
		    				ngram_range=(1,5),
		    				)
		    		),
		    		('tfidf', feature_extraction.text.TfidfTransformer(sublinear_tf=True,norm='l2')),
		    		# first SGD is for sparsity, will be tuned with alpha as large as possible...
		    		# currently not used...
		    		# ('filter',linear_model.SGDRegressor(alpha=1e-5,penalty='l1',n_iter=200)),
		    		# second SGD is for feature weighting...
					("clf",MySGDRegressor(	alpha=args.sgd_alpha, 
											penalty=args.sgd_penalty,
											learning_rate='constant',
											eta0=args.sgd_eta0,
											rho=args.sgd_rho, 
											max_iter=args.sgd_max_iter, 
											n_iter_per_step=args.sgd_n_iter_per_step))
					])



def initialize(args):
	"""
	Set everything up.
	"""
	logging.basicConfig(filename=args.logfile,mode='w',format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	train = pandas.read_table(args.trainfile,sep=',')
	leaderboard = pandas.read_table(args.testfile,sep=',')
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

		returns: ns   -- list of iteration numbers
		         aucs -- list of corresponding areas under the curve.
		"""
		y = np.array(y)
		results = [ (n, ml_metrics.auc(y,p)) for n,p in self.staged_predict(X)]

		return zip(*results) # Python idiom unzips list into two parallel ones.
		
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



class MyPipeline(pipeline.Pipeline):
	def staged_auc(self,X,y):
		"""
		MyPipeline knows about staged_auc, which 
		MySGDRegressor implements and uses.
		"""
		Xt = X
		for name, transform in self.steps[:-1]:
			Xt = transform.transform(Xt)
		return self.steps[-1][-1].staged_auc(Xt,y)










def clear_fold_info():
	for fn in os.listdir('Folds'):
		os.unlink(os.path.join('Folds',fn))


def save_fold_info(i,xs,ys):
	"""
	The fold information is stored in a simple format. These predictions are based on several 
	folds of cross-validation over the training set. 
	"""
	df = pandas.DataFrame({'iterations':xs,
							('auc%d' % i):ys})
	df.to_csv('Folds/fold%d.csv' %i,index=False)

def choose_n_iterations(show=False):
	df = None
	for fn in os.listdir('Folds'):
		fold = pandas.read_table(os.path.join('Folds',fn),sep=',',index_col='iterations')
		if isinstance(df,pandas.DataFrame):
			df = df.join(fold)
			
		else:
			df = fold

	fi = df.mean(axis=1)
	if show:
		pl.plot(fi.index,np.array(fi))
		pl.show()
	chosen = fi.index[fi.argmax()]
	logging.info("chose %d iterations, projected score %f" % (chosen,fi.max()))
	return chosen,fi.max()

NFOLDS=15


def tune_one_fold(i,train_i,test_i):
	global train
	clf = make_clf(args)
	ftrain = train[train_i]
	logging.info('fold %d' % i)
	clf.fit(ftrain.Comment,ftrain.Insult)
	ypred = clf.predict(ftrain.Comment) 
	logging.info("%d train auc=%f" % (i, ml_metrics.auc(np.array(ftrain.Insult),ypred)))
	ypred = clf.predict(train[test_i].Comment)
	# record information about the auc at each stage of training.
	xs,ys = clf.staged_auc(train[test_i].Comment,train[test_i].Insult)
	xs = np.array(xs)
	ys = np.array(ys)		
	save_fold_info(i,xs,ys)
	logging.info("saved info for fold %d" % i)

def tuning(args):
	"""
	Train the model, while holding out folds for use in
	estimating performance.

	"""
	logging.info("Tuning")
	kf = cross_validation.KFold(len(train.Insult),NFOLDS,indices=False)
	clear_fold_info()
	with ProcessPoolExecutor(max_workers=2) as executor:
		future_to_fold = dict([(executor.submit(tune_one_fold,i,train_i,test_i),i) for i,(train_i,test_i) in enumerate(kf)])
		for future in concurrent.futures.as_completed(future_to_fold):
			fold = future_to_fold[future]
			if future.exception() is not None:
				logging.warning('%r generated an exception: %s' % (fold,
                         	                            future.exception()))
			else:
				logging.info('fold %r  is finished' % (fold,))

	logging.info('tuning complete')

def get_estimates():
	if os.path.exists('Data/estimates.csv'):
		v = pandas.read_table('Data/estimates.csv',sep=',')
		return zip(v.submission,v.estimate)
	else:
		return []

def save_estimates(se):
	submissions,estimates = zip(*se)
	pandas.DataFrame(dict(submission=submissions,estimate=estimates)).to_csv('Data/estimates.csv', index=False)

def predict(args):
	"""
	Train on training file, predict on test file. 
	"""
	logging.info("Starting predictions")
	clf = make_clf(args)
	# work out how long to train for final step.
	clf.steps[-1][-1].max_iter,estimated_score = choose_n_iterations()
	clf.steps[-1][-1].reset_args()
	clf.fit(train.Comment,train.Insult)
	ypred = clf.predict(leaderboard.Comment)

	# we create a submission...
	submission = pandas.read_table('Data/sample_submission_null.csv',sep=',')
	submission['Insult'] = ypred

	if args.predictions == None:
		estimates = get_estimates()
		for x in itertools.count(1):
			filename = "submissions/submission%d.csv" % x
			if os.path.exists(filename):
				next
			else:			
				submission.to_csv(filename,index=False)
				estimates.append((filename,estimated_score))
				save_estimates(estimates)
				logging.info('Saved %s' % filename)
				break
	else:
			submission.to_csv(args.predictions,index=False)
			logging.info('Saved %s' % args.predictions)
	logging.info("Finished predictions")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Generate a prediction about insults")
	parser.add_argument('--trainfile','-T',default='Data/train.csv',help='file to train classifier on')
	parser.add_argument('--testfile','-t',default='Data/test.csv',help='file to generate predictions for')	
	parser.add_argument('--predictions','-p',default=None,help='destination for predictions (or None for default location)')
	parser.add_argument('--logfile','-l',default='insults.log',help='name of logfile')	
	parser.add_argument('--tune','-tu',action='store_true',help='if set, causes tuning step to occur')
	parser.add_argument('--sgd_alpha','-sa',type=float,default=3e-7)
	parser.add_argument('--sgd_eta0','-se',type=float,default=0.005)
	parser.add_argument('--sgd_rho','-sr',type=float,default=0.85)
	parser.add_argument('--sgd_max_iter','-smi',type=int,default=1000)
	parser.add_argument('--sgd_n_iter_per_step','-sns',type=int,default=10)
	parser.add_argument('--sgd_penalty',default="elasticnet",help='l1 or l2 or elasticnet (default: %{default}s)')
	args = parser.parse_args()
	print args
	train,leaderboard = initialize(args)
	if args.tune: tuning(args)
	predict(args)
	score.score()


