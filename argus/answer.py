# -*- coding: utf-8 -*-
"""Important objects are stored here:
    Answer - contains all information about our answer

    Source - for each found sentence, we create a source
    containing all information about the source including
    features.

    Question - all information extractable from the question alone

"""
from keyword_extract import extract
from features import Model


class Answer(object):
    def __init__(self, q):
        self.sources = []
        self.text = ''
        self.q = q
        self.info = ''
        self.model = Model(self)
        self.prob = 0


class Question(object):
    def __init__(self, question):
        self.searchwords = []
        self.not_in_kw = []
        self.text = question
        self.date_text = ''
        self.date = ''
        self.root_verb = []

        # FIXME: the extract() function modifies self in a lot of other
        # ways, actually initializing the attributes above
        self.keywords = extract(self)

        self.unknown = []
        self.query = kw_to_query(self.keywords)
        if len(self.date_text) > 0:
            self.query += ' (relevant \"' + self.date_text + '\")'


class Source():
    def __init__(self, source, url, headline, summary, sentence, date):
        self.features = []
        self.prob = 0
        self.rel = 0

        self.sentence = sentence
        self.headline = headline
        self.url = url
        self.summary = summary
        self.source = source
        self.date = date

        self.elastic = 0.


def kw_to_query(keywords):
    query = ''
    for word in keywords:
        query += word + " AND "
    query = query[:-5]
    return query
