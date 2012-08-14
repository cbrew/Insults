"""
Manage features.

1) start by tokenizing all the vocabulary, to get global frequency estimates.
"""


import pandas
import sys
import numpy as np
from collections import Counter
from enchant import tokenize
from gensim import utils  # alternative tokenizer
import logging
from sklearn import feature_selection
logging.basicConfig(filename="asap2.log",mode='w',format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
toks = tokenize.get_tokenizer('en_US')



BASIC_STOP_WORDS = set([ "the","do","it","in","to","for","and","an","are","of","is","have","a","can","could","be"])
ENGLISH_STOP_WORDS = set([
    "a", 
		"about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fify", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves"])


def ngrams(seq):
	return [(seq[i] + '_' + seq[i+1]) for i in range(len(seq)-1)]
		

def _split_enchant(s):
	words2 = utils.lemmatize(s)
	lems = [x.split('/')[0] for x in words2]
	bigs = []
	for i in range(2,len(lems)):
			bigs.append("_".join(lems[i-2:i]))
	for lem in lems:
			s = '##' + lem + '##'
			for order in range(3,9):
				for i in range(order,len(s)):
					bigs.append(s[i-order:order])
	
	
	return lems + bigs
	
splitbag = np.frompyfunc(_split_enchant,1,1)


def train_bag(text, n=500,y=None,select=False):
		"most_common is easy, but when bigrams are in the mix with unigrams, maybe not the best."
		bag = Counter()
		for document in splitbag(text):
			for word in document:
				bag[word] += 1
		"""
		bad_keys = [word for word in bag if bag[word] < 5]
		for bad_key in bad_keys:
				del bag[bad_key]
		"""
		
		if y == None or select == False: 
				logging.info("limiting %d features to %d most common" % (len(bag),n))
				return dict(bag.most_common(n))
		else:
				# might work
				logging.info("limiting %d features to %d most informative" % (len(bag),n))
				selector = feature_selection.SelectKBest(feature_selection.f_classif, k=n)		
				fea = bag_representation(bag,text)
				selector.fit(np.array(fea),y)
				chosen = selector.get_support(indices=True)
				return dict([(col,bag[col]) for col in fea.columns[chosen]])

			
def bag_count(bag,t):
		words = Counter(t)
		return [float(words[k]) for k in sorted(bag.keys())]



def bag_representation(bag, text):
			data = [bag_count(bag,t) for t in splitbag(text)]
			df = pandas.DataFrame(data,columns = sorted(bag.keys()),index=text.index)
			# we may well want to discount the larger counts...
			return df
						
