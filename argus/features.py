# -*- coding: utf-8 -*-
"""
All features are created from here.
"""
from __future__ import division
import numpy as np
from keyword_extract import nlp, verbs, extract_from_string, get_subj, get_obj
from nltk.corpus import wordnet as wn
import re
from feature_functs import tokenize, load, patterns, patterns_string
import math
import urllib2
import json

clas = '#'
rel = '@'

feature_list = ['SentimentQ', 'SentimentS', 'SubjectMatch', 'ObjectMatch', 'VerbSimSpacy',
                'VerbSimWordNet', 'RelevantDate', 'ElasticScore', 'SportScore', 'Antonyms',
                'VerbSimWordNetBinary']
feature_list_official = ['#Question Sentiment', '#Sentence Sentiment',
                         '#@Subject match', '#@Object match',
                         '#@Verb similarity (spaCy)',
                         '#@Verb similarity (WordNet)', '@Relevant date',
                         '@Elastic score', '#Sport score', '#@Antonyms',
                         '@#VerbSimWordNetBinary']


def feature_dimensions():
    w_dim = 0
    q_dim = 0
    for f in feature_list_official:
        if '#' in f:
            w_dim += 1
        if '@' in f:
            q_dim += 1
    w_dim *= 2  # XXX: must change manually
    return w_dim, q_dim


class Model(object):
    """
    RNN+ClasRel model for prediction.
    """

    def __init__(self):
        self.url = 'http://pasky.or.cz:5052/'

    def get_weights(self, nodename):
        req = urllib2.Request(self.url + 'weights/' + nodename)
        req.add_header('Content-Type','application/json')
        resp = urllib2.urlopen(req)
        rr = json.loads(resp.read())
        return np.array(rr[nodename])

    def predict(self, answer):
        s0 = ' '.join(tokenize(answer.q.text))
        s1 = [' '.join(tokenize(source.sentence)) for source in answer.sources]
        f = [source.features for source in answer.sources]
        r = {
            's0': s0,
            's1': [dict([('text', s1_)] + [
                        (feat.get_type() + feat.get_name(), feat.get_value())
                        for feat in f_
                        ])
                   for s1_, f_ in zip(s1, f)]
        }
        print(r)

        req = urllib2.Request(self.url + 'score')
        req.add_header('Content-Type', 'application/json')
        resp = urllib2.urlopen(req, json.dumps(r))
        rr = json.loads(resp.read())
        print(rr)

        return {'y': rr['score'], 'class': rr['class'], 'rel': rr['rel']}

MODEL = Model()


class Feature(object):
    """
    Represents a single feature from a single source.
    Features are inherited from this object with the exception of
    bonus features such as f==0 or multiplications.
    All features have to set its value and type in constructor.

    """

    def set_value(self, feature):
        self.feature = feature

    def set_type(self, t):
        self.type = t

    def set_info(self, info):
        self.info = info

    def set_name(self, name):
        self.name = name

    def get_value(self):
        return self.feature

    def get_type(self):
        return self.type

    def get_info(self):
        try:
            return self.info
        except AttributeError:
            return ''

    def get_name(self):
        try:
            return self.name
        except AttributeError:
            return '--feature_name--'


class ElasticScore(Feature):
    def __init__(self, answer, i):
        Feature.set_type(self, rel)
        Feature.set_name(self, 'Elastic score')
        Feature.set_value(self, answer.sources[i].elastic)


# import pickle
# import importlib
# import pysts.embedding as emb
# from argus.keras_build import config, build_model, load_sent
# module = importlib.import_module('.'+'rnn', 'models')
# conf, ps, h = config(module.config, [])
# print 'loading sts model, glove'
# glove = emb.GloVe(N=50)
# vocab = pickle.load(open('sources/vocab_sts.txt'))
# print 'glove loaded'
# model = build_model(glove, vocab, module.prep_model, conf)
# model.load_weights('sources/models/keras_model.h5')
# print 'sts model loaded'
# class STS_NN(Feature):
#     """
#     Keras models from brmson/dataset-sts
#     """
#
#     def __init__(self, answer, i):
#         Feature.set_type(self, clas + rel)
#         Feature.set_name(self, 'STS_NN')
#         gr = load_sent(answer.q.text, answer.sources[i].sentence, vocab)
#         val = model.predict(gr)['score'][:, 0][0]
#         Feature.set_value(self, val)


class SentimentQ(Feature):
    """
    Sum of emotionally colored words from the question.
    """

    def __init__(self, answer, i):
        Feature.set_type(self, clas)
        Feature.set_name(self, 'Question sentiment')
        q = sum(map(lambda word: afinn.get(word, 0), [word.lower() for word in tokenize(answer.q.text)]))
        q = float(q) / len(answer.q.text.split())
        Feature.set_value(self, q)


class SentimentS(Feature):
    """
    Sum of emotionally colored words from source sentence.
    """

    def __init__(self, answer, i):
        Feature.set_type(self, clas)
        Feature.set_name(self, 'Sentence sentiment')
        sentence = answer.sources[i].sentence
        s = sum(map(lambda word: afinn.get(word, 0), [word.lower() for word in tokenize(sentence)]))
        s = float(s) / len(sentence.split())
        Feature.set_value(self, s)


def bow(l):
    vector = np.zeros(l[0].vector.shape)
    for token in l:
        vector += token.vector
    return vector / len(l)


class RelevantDate(Feature):
    """
    1 if source date matches question date, linear decrease next 14 days, 0 otherwise.
    """

    def __init__(self, answer, i):
        Feature.set_type(self, rel)
        Feature.set_name(self, 'Date relevance')

        qdate_from, qdate_to, qdate_sloped = answer.q.date_period()
        if not qdate_from:
            Feature.set_value(self, 0.)
            return

        sdate = answer.sources[i].date
        info = 'Qdate=[%s,%s], Sdate=%s' % (qdate_from, qdate_to, sdate)
        Feature.set_info(self, info)

        if sdate < qdate_from or sdate > qdate_to:
            Feature.set_value(self, 0.)
            return
        elif qdate_sloped and (sdate - qdate_from).days > 1:
            Feature.set_value(self, 1. - (sdate - qdate_from).days / (qdate_to - qdate_from).days)
        else:
            Feature.set_value(self, 1.)


class SportScore(Feature):
    """
    Finds [num-num] in the sentence and tries to find out if subject was on the winning side.
    """

    def __init__(self, answer, i):
        Feature.set_type(self, clas)
        Feature.set_name(self, 'Match score')
        sentence = answer.sources[i].sentence
        regex = re.match('.*?(\d+)[-](\d+).*', sentence)  # (\d+)\W(\d+) only for multiple scores detection
        if regex:
            s1 = regex.group(1)
            s2 = regex.group(2)
            sentence_kw = extract_from_string(sentence)
            q = answer.q.root_verb[0]
            qsubj = get_subj(q)
            if qsubj is None:
                Feature.set_info(self, 'no q subj')
                Feature.set_value(self, 0.)
                return
            qsubj = qsubj.text
            result = 1
            if int(s1) <= int(s2):
                result = -1

            try:
                hs = load(sentence, sentence_kw, s1 + '-' + s2)
                score = patterns_string(sentence, qsubj, s1 + '-' + s2, answer.q.searchwords)
                if score == 0:
                    score = float(patterns(hs, qsubj))
                Feature.set_value(self, score * result)
                Feature.set_info(self, s1 + '-' + s2)
            except Exception:
                Feature.set_value(self, 0.)
        else:
            Feature.set_value(self, 0.)


class SubjectMatch(Feature):
    """
    1 if source sentence and question match subjects.
    """

    def __init__(self, answer, i):
        Feature.set_type(self, clas + rel)
        Feature.set_name(self, 'Subject match')
        sentence = answer.sources[i].sentence.split(':')[-1]
        q = answer.q.root_verb[0]
        qsubj = get_subj(q)
        ssubj = get_subj(list(nlp(sentence).sents)[0].root)
        if qsubj is None or ssubj is None:
            Feature.set_value(self, 0.)
            return
        info = 'Qsubject=%s, Ssubject=%s' % (qsubj.text, ssubj.text)
        Feature.set_info(self, info)
        if qsubj.lower_ in ssubj.lower_ or ssubj.lower_ in qsubj.lower_:
            Feature.set_value(self, 1.)
        else:
            Feature.set_value(self, 0.)


class ObjectMatch(Feature):
    """
    1 if question subject matches sentence object.
    """

    def __init__(self, answer, i):
        Feature.set_type(self, clas + rel)
        Feature.set_name(self, 'Object match')
        sentence = answer.sources[i].sentence.split(':')[-1]
        q = answer.q.root_verb[0]
        qsubj = get_subj(q)
        sobj = get_obj(list(nlp(sentence).sents)[0].root)

        if qsubj is None or sobj is None:
            Feature.set_value(self, 0.)
            return
        info = 'Qsubbject=%s, Sobject=%s' % (qsubj.text, sobj.text)
        Feature.set_info(self, info)
        if qsubj.lower_ in sobj.lower_ or sobj.lower_ in qsubj.lower_:
            Feature.set_value(self, 1.)
        else:
            Feature.set_value(self, 0.)


class VerbSimSpacy(Feature):
    """
    Cosine distance of question-sentence root verbs using Spacy w2v model.
    """

    def __init__(self, answer, i):
        Feature.set_type(self, clas + rel)
        Feature.set_name(self, 'Verb similarity (spacy)')
        q = answer.q
        sentence = answer.sources[i].sentence.split(':')[-1]
        doc = nlp(sentence)
        s1 = list(doc.sents)
        for j in range(len(s1)):
            if not s1[j].text.isspace():
                s_verbs = verbs(s1[j])
                break

        q_root = q.root_verb[0]
        doc = nlp(q_root.lemma_)
        s1 = list(doc.sents)
        q_root = s1[0][0]

        if len(s_verbs) == 0 or s_verbs[0].lemma_ == 'be' or q_root.lemma_ == 'be':
            Feature.set_value(self, 0.)
            return
        else:
            doc = nlp(s_verbs[0].lemma_)
            s1 = list(doc.sents)
            s_root = s1[0][0]

        q_vec = bow([q_root])
        s_vec = bow([s_root])
        info = ''
        for verb in [q_root]:
            info += 'Qverb=%s   ' % (verb)
        for verb in [s_root]:
            info += 'Sverb=%s   ' % (verb)
        Feature.set_info(self, info)
        sim = np.dot(q_vec, s_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(s_vec))
        if math.isnan(sim):
            sim = 0
        Feature.set_value(self, sim)

    #        if self.is_be(s_verbs, q.root_verb):
    #            Feature.set_value(self, 0.)

    def is_be(self, s1, s2):
        if len(s1) == 1:
            if s1[0].lemma_ == 'be':
                return True
        if len(s2) == 1:
            if s2[0].lemma_ == 'be':
                return True
        return False


class VerbSimWordNet(Feature):
    """
    Similarity of question-sentence root verbs using WordNet.
    """

    def __init__(self, answer, i):
        Feature.set_type(self, clas + rel)
        Feature.set_name(self, 'Verb similarity (WordNet)')
        q = answer.q
        sentence = answer.sources[i].sentence.split(':')[-1]
        q_verb = q.root_verb[0].lemma_
        doc = nlp(sentence)
        s1 = list(doc.sents)
        s_verb = s1[0].root.lemma_
        info = 'Qverb=%s, Sverb=%s' % (q_verb, s_verb)
        Feature.set_info(self, info)
        sim = max_sim(s_verb, q_verb)
        Feature.set_value(self, sim)


def max_sim(v1, v2):
    sim = []
    if (v1 == 'be') or (v2 == 'be'):
        return 0
    for kk in wn.synsets(v1):
        for ss in wn.synsets(v2):
            sim.append(ss.path_similarity(kk))
    if len(sim) == 0:
        return 0
    return max(0, *sim)


class VerbSimWordNetBinary(Feature):
    """
    1 if VerbSimWordNet is 1 (basically if verbs are synonyms).
    """

    def __init__(self, answer, i):
        Feature.set_type(self, clas + rel)
        Feature.set_name(self, 'Verb similarity (WordNetBinary)')
        q = answer.q
        sentence = answer.sources[i].sentence.split(':')[-1]
        q_verb = q.root_verb[0].lemma_
        doc = nlp(sentence)
        s1 = list(doc.sents)
        s_verb = s1[0].root.lemma_
        info = 'Qverb=%s, Sverb=%s' % (q_verb, s_verb)
        Feature.set_info(self, info)
        sim = max_sim(s_verb, q_verb)
        if sim == 1:
            Feature.set_value(self, sim)
        else:
            Feature.set_value(self, 0.)


class Antonyms(Feature):
    def __init__(self, answer, i):
        Feature.set_type(self, clas + rel)
        Feature.set_name(self, 'Antonyms')
        q = answer.q
        sentence = answer.sources[i].sentence
        q_verb = q.root_verb[0].lemma_
        doc = nlp(sentence)
        s1 = []
        for s in doc.sents:
            s1.append(s)
        s_verb = s1[0].root.lemma_
        sim = antonym(s_verb, q_verb)
        Feature.set_value(self, sim)


def antonym(v1, v2):
    if (v1 == 'be') or (v2 == 'be') or (v1 == v2):
        return 0
    for aa in wn.synsets(v1):
        for bb in aa.lemmas():
            if bb.antonyms():
                if v2.lower() in bb.antonyms()[0].name():
                    print v1, 'is an antonym of', v2
                    return 1
    return 0


afinn = dict(map(lambda (k, v): (k, int(v)),
                 [line.split('\t') for line in open("sources/AFINN-111.txt")]))


def zero_features(source):
    l = len(source.features)
    for i in range(l):
        f = source.features[i]
        if clas in f.get_type():
            newf = Feature()
            newf.set_type(clas)
            newf.set_name(f.get_name() + '==0')
            newf.set_value(float(f.get_value() == 0.))
            source.features.append(newf)


expand_features_list = [zero_features]  # list of used bonus-feature methods


def expand_features(answer):
    for source in answer.sources:
        for ex in expand_features_list:
            ex(source)


def gen_features(answer):
    for i in range(len(answer.sources)):
        for func in feature_list:
            answer.sources[i].features.append(eval(func)(answer, i))
    # TODO: not supported by the dataset-sts api for now
    # expand_features(answer)
