# -*- coding: utf-8 -*-
import numpy as np
from sklearn.externals import joblib
from keyword_extract import tokenize, nlp


class Features(object):  # all features for all sources
    def __init__(self):
        self.sentiments = []
        self.verb_sim = []
        self.model = joblib.load('sources/models/sentiment.pkl')
        self.prob = []

    def predict(self):
        for i in range(0,len(self.sentiments)):
            feats = np.hstack((self.sentiments[i].qs,self.verb_sim[i].sim))
#            print feats
            self.prob.append(self.model.predict_proba(feats)[0,1])

class Sentiment(object):
    def __init__(self, q, s, h):
        self.qsh = np.array([q, s, h])
        self.qs = np.array([q, s])

class Verb_Sim(object):
    def __init__(self, q, sentence):
        self.q_verb = q.root_verb[0]
        doc = nlp(sentence)
        s1 = []
        for s in doc.sents:
            s1.append(s)
        self.s_verb = s1[0].root
        self.sim = self.q_verb.similarity(self.s_verb)
#        print 'similarity: %s XXX %s = %.3f' % (self.q_verb.text, self.s_verb.text, self.sim)


afinn = dict(map(lambda (k,v): (k,int(v)),
[ line.split('\t') for line in open("sources/AFINN-111.txt") ]))

def load_verb_sim(answer):
    for i in range(len(answer.sentences)):
        answer.features.verb_sim.append(Verb_Sim(answer.q,answer.sentences[i]))

def load_sentiment(answer):
    q = sum(map(lambda word: afinn.get(word, 0), [word.lower() for word in tokenize(answer.q.text)]))
    for i in range(0,len(answer.headlines)):
        s = sum(map(lambda word: afinn.get(word, 0), [word.lower() for word in tokenize(answer.sentences[i])]))
        h = sum(map(lambda word: afinn.get(word, 0), [word.lower() for word in tokenize(answer.headlines[i])]))
        answer.features.sentiments.append(Sentiment(q,s,h))

def load_features(answer):
    load_sentiment(answer)
    load_verb_sim(answer)
