import nltk

text='was Barack Obama attacked by angry wolfes?'
#

tokens=nltk.word_tokenize(text)
print tokens

tagged = nltk.pos_tag(tokens)


propernouns = [word for word,pos in tagged if pos == 'NNP' 
or 'VB' in pos]
print tagged
print propernouns

entities = nltk.chunk.ne_chunk(tagged)
print entities