# -*- coding: utf-8 -*-
import numpy as np
from keyword_extract import tokenize, nlp, verbs
from relevance import Relevance

clas = '#'
rel = '@'

feature_list = ['Sentiment_q', 'Sentiment_s', 'Verb_sim', 'Verb_sim_wn']
feature_list_official = ['#Question Sentiment', '#Sentence Sentiment',
                         '#@Verb similarity (spaCy)',
                         '#@Verb similarity (WordNet)']
def count_flo(string):
    i = 0
    for item in feature_list_official:
        i += item.count(string)
    return i


class Features(object):  # all features for all sources
    def __init__(self):
        R = Relevance(0,0)
        R.load('sources/models')
        self.model = R
        self.features = [] # list of lists of Feature. shape=(sources, features)
        self.prob = []

    def predict(self):

        f = np.zeros((len(self.features), count_flo(clas)))
        f = []
        r = []
        r = np.zeros((len(self.features), count_flo(rel)))
        for source in self.features:
            for feat in source:
                if clas in feat.get_type():
                    f.append(feat.get_value())
                if rel in feat.get_type():
                    f.append(feat.get_value())
        try:
            f = np.array(f).reshape((len(self.features), count_flo(clas)))
            r = np.array(r).reshape((len(self.features), count_flo(rel)))
            self.prob.append(self.model.forward_propagation(f.T, r.T))
        except ValueError:
            self.prob.append(0.)

class Feature(object):  # one feature for one source

    def set_feature(self, feature):
        self.feature = feature

    def get_value(self):
        return self.feature

    def get_type(self):
        return self.type


class Sentiment_q(Feature):

    def __init__(self, answer, i):
        Feature.type = clas
        q = sum(map(lambda word: afinn.get(word, 0), [word.lower() for word in tokenize(answer.q.text)]))
        q = float(q)/len(answer.q.text.split())
        Feature.set_feature(self,q)


class Sentiment_s(Feature):
    def __init__(self, answer, i):
        Feature.type = clas
        sentence = answer.sentences[i]
        s = sum(map(lambda word: afinn.get(word, 0), [word.lower() for word in tokenize(sentence)]))
        s = float(s)/len(sentence.split())
        Feature.set_feature(self,s)

def bow(l):
    vector = np.zeros(l[0].vector.shape)
    for token in l:
        vector += token.vector
    return vector/len(l)

import math
class Verb_sim(Feature):
    def __init__(self, answer, i):
        Feature.type = clas+rel
        q = answer.q
        sentence = answer.sentences[i]
        q_vec = bow(q.root_verb)
        doc = nlp(sentence)
        s1 = []
        for s in doc.sents:
            s1.append(s)
        s_verbs = verbs(s1[0])
        s_vec = bow(s_verbs)
#        print q_vec,s_vec
        sim = np.dot(q_vec,s_vec)/(np.linalg.norm(q_vec)*np.linalg.norm(s_vec))
        if math.isnan(sim):
            sim = 0
        Feature.set_feature(self, sim)

from nltk.corpus import wordnet as wn
from keyword_extract import stop_words
class Verb_sim_wn(Feature):

    def __init__(self, answer, i):
        Feature.type = clas
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
        Feature.type = clas
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


def load_features(answer):
    for i in range(len(answer.sentences)):
        features_source = []
        for func in feature_list:
            features_source.append(eval(func)(answer, i))
        answer.features.features.append(features_source)
