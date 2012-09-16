"""
Prepare a modified version of test.csv to use as a stand-in for final.csv, which we do not have yet.

Entries are shuffled and only first 1000 entries are kept
"""

import pandas
import numpy as np

table = pandas.read_table('Data/test.csv',sep=',')
print table.tail()
index = np.array(table.index)
np.random.shuffle(index)
table = table.reindex(index)
print table.tail()
table = table.head(1000)
table.to_csv('Data/final.csv',index=False)