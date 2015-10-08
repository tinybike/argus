# -*- coding: utf-8 -*-
from keyword_extract import extract
from guardian import kw_to_query
from features import Features

class Answer(object):
    def __init__(self,q):
        self.sources = []
        self.text = ''
        self.headlines = []
        self.urls = []
        self.bodies = []
        self.sentences = []
        self.features = Features()
        self.q = q
        self.info = ''

class Question(object):
    def __init__(self,question):
        self.searchwords = []
        self.postokens = []
        self.not_in_kw = []
        self.text = question
        self.keywords = extract(self)
        self.query = kw_to_query(self.keywords)

