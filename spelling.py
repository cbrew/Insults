from enchant import tokenize
import enchant
from collections import namedtuple
import numpy as np
import pandas


toks = tokenize.get_tokenizer('en_US')
d = enchant.Dict('en_US')

Suggestion=namedtuple("Suggestion",("original","suggestions"))
Separator = namedtuple("Separator",("text"))
Corrected = namedtuple("Corrected",("text"))
Original = namedtuple("Original",("text"))



def detect(text):
	"""
	Auto spelling correct on a text.
	"""
	start = 0
	for (tok,pos) in toks(text):
		if pos > start:
			yield Separator(text[start:pos])
		if d.check(tok):
			yield tok
		else:
			yield Suggestion(tok,d.suggest(tok))
		start = pos + len(tok)
	if start < len(text):
		yield Separator(text[start:])
		
		
def suggest(original,vocabulary,suggestions):
		"""
		Make a suggestion. Choose first suggestion
		that you have seen before, otherwise first 
		suggestion.
		"""
		if suggestions == []:
			return Original(original)
		else:
			for suggestion in suggestions:
				if suggestion in vocabulary:
					return Corrected(suggestion)
			return Corrected(suggestions[0])
		
def corrections(text):
	"""
	Correct spelling in a text
	"""
	vocabulary = set()
	for token in detect(text):
		if isinstance(token,str):
			vocabulary.add(token)
			yield Original(token)
		elif isinstance(token,Suggestion):
			yield suggest(token.original,vocabulary,token.suggestions)
		else:
				yield token
				
				
def correct(text):
	return "".join(map(lambda x: x.text,corrections(text)))
	
			
pandas_correct = np.frompyfunc(correct,1,1)

	
	


if __name__ == "__main__":
	alice = """Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, `and what is the use of a book,' thought Alice `without pictures or conversation?'
	So she was considering in her own mind (as well as she could, for 
	the hot day made her feel very sleepy and stupid), whether the pleasure of 
	making a daisy-chain would be worth the trouble of getting up and picking the daisies, 
	when suddenly a White Rabbit with pink eyes ran close by her.

	There was nothing so very remarkable in that; nor did Alice thrink it so very much out of 
	the way to hear the Rabbit say to itself, `Oh dear! Oh dear! I shall be late!' 
	(when she thought it over afterwards, it occurred to her that she ought to have wondered 
	at this, but at the time it all seemed quite natural); but when the Rabbit actually took a 
	watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started 
	to her feet, for it flashed across her mind that she had never before seen a 
	rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with 
	curiosity, she ran across the field after it, and fortunately was just in time to see 
	it pop down a large rabbig-hole under the hrdge.
	"""
	print correct(alice)
	
	print "Correcting spelling"
	v = pandas.read_table('Data/train.csv',sep=',')
	v.Comment = pandas_correct(v.Comment)
	v.to_csv("Data/train_corrected.csv",header=True,index=False)
	v = pandas.read_table('Data/test.csv',sep=',')
	v.Comment = pandas_correct(v.Comment)
	v.to_csv("Data/test_corrected.csv",header=True,index=False)
	print "Spelling corrected"
