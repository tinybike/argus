# -*- coding: utf-8 -*-
import numpy as np
from sklearn.externals import joblib
from keyword_extract import tokenize, nlp



class Features(object):  # all features for all sources
    def __init__(self):
        self.model = joblib.load('sources/models/sentiment.pkl')
        self.features = [] # list of lists of Feature. shape=(sources, features)
        self.prob = []

    def predict(self):
        feats = np.zeros((len(self.features), len(self.features[0])))
        for i in range(len(self.features)):
            for j in range(len(self.features[0])):
                feats[i][j] = self.features[i][j].get_feature()
            if len(self.model.coef_[0]) != len(feats[0]):
                self.prob.append(0.)
            else:
                self.prob.append(self.model.predict_proba(feats[i])[0,1])



class Feature(object):  # one feature for one source

    def set_feature(self, feature):
        self.feature = feature

    def get_feature(self):
        return self.feature


class Sentiment_q(Feature):

    def __init__(self, answer, i):
        q = sum(map(lambda word: afinn.get(word, 0), [word.lower() for word in tokenize(answer.q.text)]))
        q = float(q)/len(answer.q.text.split())
        Feature.set_feature(self,q)


class Sentiment_s(Feature):
    def __init__(self, answer, i):
        sentence = answer.sentences[i]
        s = sum(map(lambda word: afinn.get(word, 0), [word.lower() for word in tokenize(sentence)]))
        s = float(s)/len(sentence.split())
        Feature.set_feature(self,s)


class Verb_sim(Feature):
    def __init__(self, answer, i):
        q = answer.q
        sentence = answer.sentences[i]
        self.q_verb = q.root_verb[0]
        doc = nlp(sentence)
        s1 = []
        for s in doc.sents:
            s1.append(s)
        self.s_verb = s1[0].root
        self.sim = self.q_verb.similarity(self.s_verb)
        Feature.set_feature(self,self.sim)
#        print 'similarity: %s XXX %s = %.3f' % (self.q_verb.text, self.s_verb.text, self.sim)

from nltk.corpus import wordnet as wn
from keyword_extract import stop_words
class Verb_sim_wn(Feature):

    def __init__(self, answer, i):
        q = answer.q
        sentence = answer.sentences[i]
        q_verb = q.root_verb[0].lemma_
        doc = nlp(sentence)
        s1 = []
        for s in doc.sents:
            s1.append(s)
        s_verb = s1[0].root.lemma_
        sim = self.max_sim(s_verb, q_verb)
#        print 'question=',q.text
#        print 'sentence=',sentence
#        print 'matching',q_verb,s_verb,'score=',sim
#        print '-----------------------------------------'
        Feature.set_feature(self, sim)

    def max_sim(self, v1, v2):
        sim = []
        if (v1 == 'be') or (v2 == 'be'):
            return 0
        for kk in wn.synsets(v1):
            for ss in wn.synsets(v2):
                sim.append(ss.path_similarity(kk))
        if len(sim) == 0:
            return 0
        return max(0, *sim)

class Antonyms(Feature):

    def __init__(self, answer, i):
        q = answer.q
        sentence = answer.sentences[i]
        q_verb = q.root_verb[0].lemma_
        doc = nlp(sentence)
        s1 = []
        for s in doc.sents:
            s1.append(s)
        s_verb = s1[0].root.lemma_
        sim = self.antonym(s_verb, q_verb)
        Feature.set_feature(self, sim)

    def antonym(self, v1, v2):
        for aa in wn.synsets(v1):
            for bb in aa.lemmas():
                if bb.antonyms():
                    try:
                        if v2.lower in bb.antonyms()[0].name():
                            print v1, 'is an antonym of', v2
                            return 1
                    except TypeError:
                        continue
        return 0


afinn = dict(map(lambda (k,v): (k,int(v)),
[ line.split('\t') for line in open("sources/AFINN-111.txt") ]))

feature_list = ['Sentiment_q', 'Sentiment_s', 'Verb_sim', 'Verb_sim_wn']
feature_list_official = ['Question Sentiment', 'Sentence Sentiment',
                         'Verb similarity (spaCy)', 'Verb similarity (WordNet)']


def load_features(answer):
    for i in range(len(answer.sentences)):
        features_source = []
        for func in feature_list:
            features_source.append(eval(func)(answer, i))
        answer.features.features.append(features_source)
