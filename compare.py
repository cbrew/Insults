"""
This code suggests that submission80 is very like 57
"""



import pandas
import sys

sub1 = pandas.read_table(sys.argv[1],sep=',')
df = pandas.DataFrame({sys.argv[1]:sub1.Insult})
for arg in sys.argv[2:]:
	sub2 = pandas.read_table(arg,sep=',')

	df[arg]=sub2.Insult

print df.corr().to_string()