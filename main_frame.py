from guardian import get_content
from keyword_extract import check_keywords, tokenize
from answer import Question, Answer
from sklearn import linear_model
import numpy as np
from sklearn.externals import joblib
#from nltk.corpus import sentiwordnet as swn
#import nltk


def get_answer(question):
    a = Answer(Question(question))

    checked = check_keywords(a.q)

    if not checked:
        a.q.query += ' ('+str(a.q.not_in_kw)+'not in keywords)'
        a.text = 'Didn\'t understand the question'
        return a

    found = get_content(a)
    if found:
        a.text = sentiment_learned(a)
        return a

    if 'bsolutely' in a.headline:
        a.text = 'Absolutely not sure'
    else:
        a.text = 'Not sure'
    return a

afinn = dict(map(lambda (k,v): (k,int(v)),
                     [ line.split('\t') for line in open("sources/AFINN-111.txt") ]))

def sentiment(answer):
    ans = ''
    q = sum(map(lambda word: afinn.get(word, 0), [word.lower() for word in tokenize(answer.q.text)]))
    s = sum(map(lambda word: afinn.get(word, 0), [word.lower() for word in tokenize(answer.sentence)]))
    h = sum(map(lambda word: afinn.get(word, 0), [word.lower() for word in tokenize(answer.headline)]))
    answer.sentiment = [str(q), str(s), str(h)]

    a = s + h
    if q == 0:
        if a < 0:
            ans = 'NO'
        else:
            ans = 'YES'
    elif q > 0:
        if a < 0:
            ans = 'NO'
        else:
            ans = 'YES'
    elif q < 0:
        if a < 0:
            ans = 'NO'
        else:
            ans = 'YES'
    return ans

def sentiment_learned(answer):
    q = sum(map(lambda word: afinn.get(word, 0), [word.lower() for word in tokenize(answer.q.text)]))
    s = sum(map(lambda word: afinn.get(word, 0), [word.lower() for word in tokenize(answer.sentence)]))
    h = sum(map(lambda word: afinn.get(word, 0), [word.lower() for word in tokenize(answer.headline)]))
    answer.sentiment = [str(q), str(s), str(h)]
    clf = joblib.load('sources/models/sentiment.pkl')
    x = np.array([q, s, h])
    a=clf.predict_proba(x)[:,1]
    if a < 0.5:
        return 'NO'
    return 'YES'

#text = ''



#print sum(map(lambda word: afinn.get(word, 0), text.lower().split()))










