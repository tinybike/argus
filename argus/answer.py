# -*- coding: utf-8 -*-
from keyword_extract import extract
from features import Features


class Answer(object):
    def __init__(self, q):
        self.sources = []
        self.text = ''
        self.headlines = []
        self.urls = []
        self.bodies = []
        self.sentences = []
        self.features = Features()
        self.q = q
        self.info = ''
        self.elastic = []


class Question(object):
    def __init__(self, question):
        self.searchwords = []
        self.postokens = []
        self.not_in_kw = []
        self.text = question
        self.date_text = ''
        self.date = ''
        self.root_verb = []
        self.keywords = extract(self)
        self.unknown = []
#        print '>>>>>>>>>>>>>>'
#        print self.text
#        print self.keywords
#        print '<<<<<<<<<<<<<<'
        self.query = kw_to_query(self.keywords)
        if len(self.date_text) > 0:
            self.query += ' (relevant \"'+self.date_text+'\")'


def kw_to_query(keywords):
    query = ''
    for word in keywords:
            query += word + " AND "
    query = query[:-5]
    return query
