# -*- coding: utf-8 -*-
import numpy as np
from sklearn.externals import joblib
from keyword_extract import tokenize


class Features(object):  # all features for all sources
    def __init__(self):
        self.sentiments = []
        self.model = joblib.load('sources/models/sentiment.pkl')
        self.prob = []

    def predict(self):
        for i in range(0,len(self.sentiments)):
            self.prob.append(self.model.predict_proba(self.sentiments[i].qsh)[0,1])

class Sentiment(object):
    def __init__(self,q,s,h):
        self.qsh = np.array([q,s,h])

afinn = dict(map(lambda (k,v): (k,int(v)),
                     [ line.split('\t') for line in open("sources/AFINN-111.txt") ]))

def load_sentiment(answer):
    q = sum(map(lambda word: afinn.get(word, 0), [word.lower() for word in tokenize(answer.q.text)]))
    for i in range(0,len(answer.headlines)):
        s = sum(map(lambda word: afinn.get(word, 0), [word.lower() for word in tokenize(answer.sentences[i])]))
        h = sum(map(lambda word: afinn.get(word, 0), [word.lower() for word in tokenize(answer.headlines[i])]))
        answer.features.sentiments.append(Sentiment(q,s,h))
