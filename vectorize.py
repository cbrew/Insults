import pandas
from sklearn import feature_extraction,linear_model,cross_validation,ensemble,svm,pipeline,grid_search
import ml_metrics
import numpy as np
import os
import itertools
import logging

logging.basicConfig(filename="vectorize.log",mode='w',format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
train = pandas.read_table('Data/train.csv',sep=',')
leaderboard = pandas.read_table('Data/test.csv',sep=',')

class MyLogisticRegression(linear_model.LogisticRegression):
    def predict(self, X):
        return linear_model.LogisticRegression.predict_proba(self, X)[:,1]




pipeline = pipeline.Pipeline([
    ('vect', feature_extraction.text.CountVectorizer(lowercase=True,max_n=2)),
    ('tfidf', feature_extraction.text.TfidfTransformer()),
    ('clf', MyLogisticRegression(tol=1e-3)),
])

parameters = {
# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
# 'vect__max_df': (0.5, 0.75, 1.0),
    # 'vect__analyzer':('word','char'),
    #  'vect__lowercase': (True,False),
	
    # 'vect__max_features': (None, 10, 50, 100, 500,1000,5000, 10000, 50000),
    'vect__max_n': (1, 2),  # words or bigrams
#    'tfidf__use_idf': (True, False),
#    'tfidf__norm': ('l1', 'l2'),
    'clf__C': (1,5,25,35,45,50,75,80,85,90,100),
    # 'clf__tol': (1e-3,1e-4),
    # 'clf__penalty': ("l1","l2"),
    'clf__class_weight': ("auto",None),
		    
# 'clf__alpha': (0.00001, 0.000001),
# 'clf__penalty': ('l2', 'elasticnet'),
#    'clf__n_iter': (10, 50, 80),
}


clf =  grid_search.GridSearchCV(pipeline, parameters, n_jobs=2,score_func=ml_metrics.auc,verbose=2)  
# XXX add score_func=ml_metrics.auc, which requires a custom version of LogisticRegression that
# predicts using predict_proba


			
ss = 0
n = 0
kf = cross_validation.ShuffleSplit(len(train),5,train_size=0.9,indices=False,random_state=1234)
for i,(train_i,test_i) in enumerate(kf):
	n_samples = train_i.sum()
	logging.info('fold %d' % i)
	clf.fit(train[train_i].Comment,train[train_i].Insult)
	best_parameters = clf.best_estimator_.get_params()
	for param_name in sorted(parameters.keys()):
		logging.info("\t%d %s: %r" % (i,param_name, best_parameters[param_name]))
	ypred = clf.predict(train[train_i].Comment) 
	logging.info("%d train=%f" % (i, ml_metrics.auc(np.array(train[train_i].Insult),ypred)))
	ypred = clf.predict(train[test_i].Comment)
	est = ml_metrics.auc(np.array(train[test_i].Insult),ypred)
	logging.info("test %f" % est)
	ss += est
	n  += 1
logging.info('Expected score %f' % (ss/n))
logging.info("Starting leaderboard")
clf.fit(train.Comment,train.Insult)
best_parameters = clf.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
	logging.info("\tL %s: %r" % (param_name, best_parameters[param_name]))
ypred = clf.predict_proba(leaderboard.Comment)[:,1]
submission = pandas.read_table('Data/sample_submission_null.csv',sep=',')
submission['Insult'] = ypred
for x in itertools.count(1):
		filename = "submissions/submission%d.csv" % x
		if os.path.exists(filename):
			next
		else:
			
			submission.to_csv(filename,index=False)
			logging.info("saved to %s" % filename)
			break
print "Done"
