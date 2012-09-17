"""
Code for the insults competition run by Kaggle for Impermium in August-September 2012.

Prerequisites
-------------

To run the script at all we need.

Python 2.7.3 + scikit-learn 0.13-git + ml_metrics 0.1.1 + pandas 0.8.1 + concurrent.futures

 (+ matplotlib 1.2.x for plotting, but not needed to predict)
(all have licenses that permit commercial use)

To make tracking of experiments easy, we use Sumatra 


Hardware
--------

I ran on a four year old iMac with 4Gb of memory and dual core Intel processor.

Sumatra requires the datafiles in ./Data


Ideas
-----

- use character n-grams because they are robust and simple.
- tune SGD carefully.


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



# Dataflow
# --------
#
# training file + parameters -> folds -> chosen parameters + performance estimates
# training file + chosen parameters -> model
# labeled test file + model -> predictions -> score
# unlabeled test file + model -> predictions

# Storage plan
# ------------
# Everything should be created inside a directory called Data, because that is what
# Sumatra (packages.python.org/Sumatra) expects.


def DataFile(category,name):
	"""
	Create a filename within Data
	"""
	return os.path.join('Data',category,name)

def DataDirectory(category):
	return os.path.join('Data',category)


def LogFile(name):
	return os.path.join('Logs',name)

	
class MySGDRegressor(linear_model.SGDRegressor):
	"""
	A customized SGD regressor that 
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
	"""
	Custom version of scikit-learn's Pipeline class, with an extra method for 
	use in tuning.
	"""
	def staged_auc(self,X,y):
		"""
		MyPipeline knows about staged_auc, which 
		MySGDRegressor implements and uses.
		"""
		Xt = X
		for name, transform in self.steps[:-1]:
			Xt = transform.transform(Xt)
		return self.steps[-1][-1].staged_auc(Xt,y)


def get_estimates():
	estimate_file = DataFile('Estimates','estimates.csv')
	if os.path.exists(estimate_file):
		v = pandas.read_table(estimate_file,sep=',')
		return zip(v.submission,v.estimate)
	else:
		return []

def save_estimates(se):
	estimate_file = DataFile('Estimates','estimates.csv')
	submissions,estimates = zip(*se)
	pandas.DataFrame(dict(submission=submissions,estimate=estimates)).to_csv(estimate_file, index=False)


def make_full_training():
	df1 = pandas.read_table(DataFile('Inputs','train.csv'),sep=',')
	df2 = pandas.read_table(DataFile('Inputs','test_with_solutions.csv'),sep=',')
	df = pandas.concat([df1,df2])
	df.to_csv(DataFile('Inputs','fulltrain.csv'),index=False)




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





def scale_predictions(ypred):
	"""
	normalize range of predictions to 0-1. 
	"""

	yrange = ypred.max() - ypred.min()
	ypred -= ypred.min()
	ypred /= yrange
	
	# protection against rounding, ENSURE nothing out of range.
	ypred[ypred > 1] = 1
	ypred[ypred < 0] = 0
	return ypred

def clear_fold_info():
	"""
	Clear the information about folds. This is created during the tuning step.
	"""
	for fn in os.listdir(DataDirectory('Folds')):
		os.unlink(DataFile('Folds',fn))


def save_fold_info(i,xs,ys):
	"""
	The fold information is stored in a simple format. These predictions are based on several 
	folds of cross-validation over the training set. 
	"""
	df = pandas.DataFrame({'iterations':xs,
							('auc%d' % i):ys})
	df.to_csv(DataFile('Folds','fold%d.csv' %i),index=False)

def choose_n_iterations(show=False):
	"""
	work out how many iterations to use, using data stashed during tuning.
	"""
	df = None
	for fn in os.listdir(DataDirectory('Folds')):
		fold = pandas.read_table(DataFile('Folds',fn),sep=',',index_col='iterations')
		if isinstance(df,pandas.DataFrame):
			df = df.join(fold)
			
		else:
			df = fold

	if len(df.columns) == 0:
		logging.error('cannot determine number of iterations until tune step has been done')
		sys.exit()

	fi = df.mean(axis=1)
	if show:
		pl.plot(fi.index,np.array(fi))
		pl.show()
	chosen = fi.index[fi.argmax()]
	logging.info("chose %d iterations, projected score %f" % (chosen,fi.max()))
	return chosen,fi.max()




def tune_one_fold(i,train_i,test_i):
	"""
	Tune one fold of the data.
	"""
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



NFOLDS=15

def initialize(args):
	"""
	Set up the training and test data.
	"""
	train = pandas.read_table(args.trainfile,sep=',')
	leaderboard = pandas.read_table(args.testfile,sep=',')
	return train,leaderboard

def tuning(args):
	"""
	Train the model, while holding out folds for use in
	estimating performance.

	"""
	logging.info("Tuning")
	kf = cross_validation.KFold(len(train.Insult),NFOLDS,indices=False)
	clear_fold_info()

	# if we had more than 2 cores, max_workers could be > 2. For now 2 is comfortable...
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
	submission = pandas.DataFrame(
			dict(Insult=ypred,Comment=leaderboard.Comment,Date=leaderboard.Date),
			columns=('Insult','Date','Comment'))

	if args.predictions == None:
		estimates = get_estimates()
		for x in itertools.count(1):
			filename = DataFile("Submissions","submission%d.csv" % x)
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


def run_prediction(parser=None,args_in=None,competition=False):
	"""
	Either pick up the arguments from the command line or use the 
	ones pre-packaged for the script.
	"""
	global train
	global leaderboard

	if competition:
		logging.info('Running prepackaged arguments (%r)' % args_in)
		args = parser.parse_args(args_in)
	else:
		logging.info('Using arguments from command line %r' % args_in)
		args = args_in

	train,leaderboard = initialize(args)
	if args.tune: tuning(args)
	predict(args)
	if args.score:
		score.score()




if __name__ == "__main__":
	competition_argsets = (
		[
			"--tune",
			"--sgd_alpha","1e-5",
			"--trainfile",DataFile('Inputs',"fulltrain.csv"),
			"--testfile",DataFile('Inputs',"final.csv"),
			'--predictions',DataFile('Final','final1.csv'),
			'--no_score'],

		[
			"--tune",
			"--sgd_alpha","1e-6",
			"--trainfile",DataFile('Inputs',"fulltrain.csv"),
			"--testfile",DataFile('Inputs',"final.csv"),
			'--predictions',DataFile('Final','final2.csv'),
			'--no_score'],
		[
			"--tune",
			"--sgd_alpha","5e-7",
			"--trainfile",DataFile('Inputs',"fulltrain.csv"),
			"--testfile",DataFile('Inputs',"final.csv"),
			'--predictions',DataFile('Final','final3.csv'),
			'--no_score'],
		[
			"--tune",
			"--sgd_alpha","5e-6",
			"--trainfile",DataFile('Inputs',"fulltrain.csv"),
			"--testfile",DataFile('Inputs',"final.csv"),
			'--predictions',DataFile('Final','final4.csv'),
			'--no_score'],
		[
			"--tune",
			"--sgd_alpha","1e-7",
			"--trainfile",DataFile('Inputs',"fulltrain.csv"),
			"--testfile",DataFile('Inputs',"final.csv"),
			'--predictions',DataFile('Final','final5.csv'),
			'--no_score'],
			)



	parser = argparse.ArgumentParser(description="Generate a prediction about insults")
	parser.add_argument('--trainfile','-T',default=DataFile('Inputs','train.csv'),help='file to train classifier on')
	parser.add_argument(
							'--testfile','-t',
							default=DataFile('Inputs','test.csv'),
							help='file to generate predictions for'
						)	
	parser.add_argument('--predictions','-p',default=None,help='destination for predictions (or None for default location)')
	parser.add_argument('--logfile','-l',
						default=LogFile('insults.log'),
						help='name of logfile'
						)	
	parser.add_argument('--tune','-tu',
						action='store_true',
						help='if set, causes tuning step to occur'
						)

	# linear classifier parameters
	parser.add_argument('--sgd_alpha','-sa',type=float,default=1e-5)
	parser.add_argument('--sgd_eta0','-se',type=float,default=0.005)
	parser.add_argument('--sgd_rho','-sr',type=float,default=0.999)
	parser.add_argument('--sgd_max_iter','-smi',type=int,default=1000)
	parser.add_argument('--sgd_n_iter_per_step','-sns',type=int,default=20)
	parser.add_argument('--sgd_penalty','-sp',default="elasticnet",help='l1 or l2 or elasticnet (default: %{default}s)')

	# other parameters.

	parser.add_argument('--competition','-c',action='store_true',help='make predictions for the final stage of the competition')
	parser.add_argument('--score','-sc',action='store_true',dest='score',help='turn on print out of score at end', default=True)
	parser.add_argument('--no_score','-nsc',action='store_false',dest='score',help='turn off print out of score at end' )


	# this code is designed to play nice with Sumatra, which likes a single argument on command line
	# pointing to a config file. So we make a config file containing command lines that we might otherwise 
	# have typed. A bit of tinkering is needed to make Sumatra like it.
	# use of execfile is not nice, but Sumatra is useful, so...ok (and this is a thin wrapper round command line anyway)
	parameters={}
	execfile(sys.argv[1],parameters)
	
	for line in parameters['commands']:
		args = parser.parse_args(line)
		if args.competition:
			logging.basicConfig(filename=LogFile('final.log'),mode='w',format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
			for argset in competition_argsets:
				run_prediction(parser=parser,args_in=argset,competition=True)
		else:
			logging.basicConfig(filename=args.logfile,mode='w',format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
			run_prediction(parser=parser,args_in=args,competition=False)






