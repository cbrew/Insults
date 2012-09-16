"""
Code to track performance. Should really pick up the 
estimate derived from cross-validation too.
"""

import pandas
import ml_metrics
import os
import numpy as np
from insults import DataFile

def score():
	gold = pandas.read_table(DataFile('Inputs','test_with_solutions.csv'),sep=',')
	private = gold[gold.Usage=='PrivateTest'].Insult
	public = gold[gold.Usage=='PublicTest'].Insult
	data = []
	for fn in os.listdir('submissions'):
			if fn[-4:] == ".csv":
				guess = pandas.read_table(os.path.join('submissions',fn),sep=',')
				pub_guess = guess.Insult[public.index]
				priv_guess = guess.Insult[private.index]
				data.append({"fn": fn[:-4],
									"score" :ml_metrics.auc(gold.Insult,guess.Insult),
									"public": ml_metrics.auc(np.array(public),np.array(pub_guess)),
									"private": ml_metrics.auc(np.array(private),np.array(priv_guess)),
									})

	print pandas.DataFrame(data,columns=("fn","score","public","private")).sort('score')

if __name__ == "__main__":
		score()