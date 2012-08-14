import pandas
from sklearn import feature_extraction,linear_model,cross_validation
import ml_metrics
import numpy as np

train = pandas.read_table('Data/train.csv',sep=',')

cv = feature_extraction.text.CountVectorizer()

kf = cross_validation.KFold(len(train),5,indices=False)
for i,(train_i,test_i) in enumerate(kf):
	n_samples = train_i.sum()
	w = cv.fit_transform(train[train_i].Comment)
	clf = linear_model.LogisticRegression()
	clf.fit(w,train[train_i].Insult)
	ypred = clf.predict_proba(w)[:,1]
	print i, ml_metrics.auc(ypred,np.array(train[train_i].Insult)),
	w = cv.transform(train[test_i].Comment)
	ypred = clf.predict_proba(w)[:,1]
	print ml_metrics.auc(ypred,np.array(train[test_i]))

