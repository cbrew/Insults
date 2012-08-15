import pandas
from sklearn import feature_extraction,linear_model,cross_validation,ensemble,svm,pipeline,grid_search
import ml_metrics
import numpy as np
import os
import itertools
import logging

train = pandas.read_table('Data/train.csv',sep=',')
leaderboard = pandas.read_table('Data/test.csv',sep=',')



pipeline = pipeline.Pipeline([
    ('vect', feature_extraction.text.CountVectorizer(lowercase=True,max_n=2)),
    ('tfidf', feature_extraction.text.TfidfTransformer()),
    ('clf', linear_model.LogisticRegression(fit_intercept=True,class_weight="auto")),
])

parameters = {
# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
# 'vect__max_df': (0.5, 0.75, 1.0),
    # 'vect__analyzer':('word','char'),
    # 'vect__max_features': (None, 10, 50, 100, 500,1000,5000, 10000, 50000),
    'vect__max_n': (1, 2,3),  # words or bigrams
#    'tfidf__use_idf': (True, False),
#    'tfidf__norm': ('l1', 'l2'),
    'clf__C': (10,20,40,60,70,80,100,160,320),
# 'clf__alpha': (0.00001, 0.000001),
# 'clf__penalty': ('l2', 'elasticnet'),
#    'clf__n_iter': (10, 50, 80),
}


clf =  grid_search.GridSearchCV(pipeline, parameters, n_jobs=2)


			
ss = 0
n = 0
kf = cross_validation.ShuffleSplit(len(train),10,train_size=0.9,indices=False,random_state=1234)
for i,(train_i,test_i) in enumerate(kf):
	n_samples = train_i.sum()
	
	clf.fit(train[train_i].Comment,train[train_i].Insult)
	best_parameters = clf.best_estimator_.get_params()
	for param_name in sorted(parameters.keys()):
		print "\t%s: %r" % (param_name, best_parameters[param_name])
	ypred = clf.predict_proba(train[train_i].Comment) [:,1]
	print i, ml_metrics.auc(np.array(train[train_i].Insult),ypred),
	ypred = clf.predict_proba(train[test_i].Comment) [:,1]
	est = ml_metrics.auc(np.array(train[test_i].Insult),ypred)
	print est
	ss += est
	n  += 1
print ss/n
print "Leaderboard"
clf.fit(train.Comment,train.Insult)
best_parameters = clf.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
	print "\t%s: %r" % (param_name, best_parameters[param_name])
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
