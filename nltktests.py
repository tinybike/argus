import nltk

text='Were the New England Patriots going to be called the Bay State Patriots?'
#

tokens=nltk.word_tokenize(text)
print tokens

tagged = nltk.pos_tag(tokens)


propernouns = [word for word,pos in tagged if pos == 'NNP' 
or 'VB' in pos or 'CD' in pos]
print tagged
print propernouns

entities = nltk.chunk.ne_chunk(tagged)
print entities

