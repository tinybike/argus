# -*- coding: utf-8 -*-
import numpy as np
from keyword_extract import tokenize, nlp, verbs
from relevance import Relevance
from nltk.corpus import wordnet as wn

clas = '#'
rel = '@'

feature_list = ['Sentiment_q', 'Sentiment_s', 'Subj_match', 'Obj_match', 'Verb_sim',
                'Verb_sim_wn', 'Relevant_date']
feature_list_official = ['#Question Sentiment', '#Sentence Sentiment',
                         '#@Subject match','#@Object match',
                         '#@Verb similarity (spaCy)',
                         '#@Verb similarity (WordNet)', '@Relevant date']
def count_flo(string):
    i = 0
    for item in feature_list_official:
        i += item.count(string)
    return i


class Model(object):  # all features for all sources
    def __init__(self, answer):
        R = Relevance(0, 0)
        R.load('sources/models')
        self.model = R
        self.ansprob = 0
        self.answer = answer

    def predict(self):
        f = []
        r = []
        for source in self.answer.sources:
            for feat in source.features:
                if clas in feat.get_type():
                    f.append(feat.get_value())
                if rel in feat.get_type():
                    r.append(feat.get_value())
        try:
            f = np.array(f).reshape((len(self.answer.sources), count_flo(clas)))
            r = np.array(r).reshape((len(self.answer.sources), count_flo(rel)))
            self.answer.prob = self.model.forward_propagation(f.T, r.T)
            probs, rels = self.model.probs_rels(f.T, r.T)
            for i in range(len(self.answer.sources)):
                self.answer.sources[i].prob = probs[i]
                self.answer.sources[i].rel = rels[i]
        except ValueError:
            self.ansprob = 0.

class Feature(object):  # one feature for one source

    def set_value(self, feature):
        self.feature = feature

    def set_type(self, t):
        self.type = t

    def set_info(self, info):
        self.info = info

    def get_value(self):
        return self.feature

    def get_type(self):
        return self.type

    def get_info(self):
        try:
            return self.info
        except AttributeError:
            return ''


class Sentiment_q(Feature):
    def __init__(self, answer, i):
        Feature.set_type(self, clas)
        q = sum(map(lambda word: afinn.get(word, 0), [word.lower() for word in tokenize(answer.q.text)]))
        q = float(q)/len(answer.q.text.split())
        Feature.set_value(self,q)


class Sentiment_s(Feature):
    def __init__(self, answer, i):
        Feature.set_type(self, clas)
        sentence = answer.sources[i].sentence
        s = sum(map(lambda word: afinn.get(word, 0), [word.lower() for word in tokenize(sentence)]))
        s = float(s)/len(sentence.split())
        Feature.set_value(self, s)

def bow(l):
    vector = np.zeros(l[0].vector.shape)
    for token in l:
        vector += token.vector
    return vector/len(l)

import math
from dateutil.parser import parse

class Relevant_date(Feature):
    def __init__(self, answer, i):
        Feature.set_type(self, rel)
        sdate = answer.sources[i].date
        qdate = answer.q.date
#        print type(sdate), type(qdate)
        try:
            sdate = parse(sdate, ignoretz=True, fuzzy=True).date()
            qdate = parse(qdate, ignoretz=True, fuzzy=True).date()
            info = sdate + qdate
#            print sdate, qdate
            delta = qdate-sdate
            f = self.gauss(delta.days)
#            print 'gaus(timedelta) =', f
            Feature.set_value(self, f)
            Feature.set_info(self, info)
        except TypeError:
            Feature.set_value(self, 0.)

    def gauss(self, x):
        mu = 2
        delta = math.sqrt(3)
        return math.exp(-(x-mu)**2/(2*delta**2))


class Verb_sim(Feature):
    def __init__(self, answer, i):
        Feature.set_type(self, clas+rel)
        q = answer.q
        sentence = answer.sources[i].sentence
        q_vec = bow(q.root_verb)
        doc = nlp(sentence)
        s1 = []
        for s in doc.sents:
            s1.append(s)
        s_verbs = verbs(s1[0])
        s_vec = bow(s_verbs)
        sim = np.dot(q_vec,s_vec)/(np.linalg.norm(q_vec)*np.linalg.norm(s_vec))
        if math.isnan(sim):
            sim = 0
        Feature.set_value(self, sim)

class Subj_match(Feature):
    def __init__(self, answer, i):
        Feature.set_type(self, clas+rel)
        sentence = answer.sources[i].sentence
        q = answer.q.root_verb[0]
        qsubj = self.get_subj(q)
        ssubj = self.get_subj(list(nlp(sentence).sents)[0].root)
        if qsubj is None or ssubj is None:
            Feature.set_value(self, 0.)
            return
        info = 'Qsubject=%s, Ssubject=%s' % (qsubj.text, ssubj.text)
        Feature.set_info(self, info)
        if qsubj.lower_ in ssubj.lower_ or ssubj.lower_ in qsubj.lower_:
            Feature.set_value(self, 1.)
        else:
            Feature.set_value(self, 0.)

    def get_subj(self, root):
        for child in root.children:
            if child.dep_ == 'nsubj':
                return child


class Obj_match(Feature):
    def __init__(self, answer, i):
        Feature.set_type(self, clas+rel)
        sentence = answer.sources[i].sentence
        q = answer.q.root_verb[0]
        qsubj = self.get_subj(q)
        sobj = self.get_obj(list(nlp(sentence).sents)[0].root)

        if qsubj is None or sobj is None:
            Feature.set_value(self, 0.)
            return
        info = 'Qsubbject=%s, Sobject=%s' % (qsubj.text, sobj.text)
        Feature.set_info(self, info)
        if qsubj.lower_ in sobj.lower_ or sobj.lower_ in qsubj.lower_:
            Feature.set_value(self, 1.)
        else:
            Feature.set_value(self, 0.)

    def get_subj(self, root):
        for child in root.children:
            if child.dep_ == 'nsubj':
                return child
    def get_obj(self, root):
        for child in root.children:
            if child.dep_ == 'dobj':
                return child


class Verb_sim_wn(Feature):
    def __init__(self, answer, i):
        Feature.set_type(self, clas+rel)
        q = answer.q
        sentence = answer.sources[i].sentence
        q_verb = q.root_verb[0].lemma_
        doc = nlp(sentence)
        s1 = []
        for s in doc.sents:
            s1.append(s)
        s_verb = s1[0].root.lemma_
        info = 'Qverb=%s, Sverb=%s' % (q_verb, s_verb)
        Feature.set_info(self, info)
        sim = self.max_sim(s_verb, q_verb)
        Feature.set_value(self, sim)

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
        Feature.set_type(self, clas)
        q = answer.q
        sentence = answer.sentences[i]
        q_verb = q.root_verb[0].lemma_
        doc = nlp(sentence)
        s1 = []
        for s in doc.sents:
            s1.append(s)
        s_verb = s1[0].root.lemma_
        sim = self.antonym(s_verb, q_verb)
        Feature.set_value(self, sim)

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
    for i in range(len(answer.sources)):
#        features_source = []
        for func in feature_list:
            answer.sources[i].features.append(eval(func)(answer, i))
#        answer.features.features.append(features_source)
