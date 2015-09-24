# -*- coding: utf-8 -*-
from keyword_extract import extract
from guardian import kw_to_query

class Answer(object):
    q =[]
    text = ''
    headline = ''
    url = ''
    body = ''
    sentence = ''
    sources = ('', '', '', '')
    def __init__(self,q):
        self.q = q
    def set_sources(self):
        (self.headline, self.url, self.body, self.sentence) = self.sources


class Question(object):
    text = ''
    keywords = []
    searchwords = []
    postokens = []
    query = ''
    def __init__(self,question):
        self.text = question
        self.keywords = extract(self)
        self.query = kw_to_query(self.searchwords)



#ans = Answer(Question('Will the Chinese Stock Market continue to rise throughout 2015?'))
##
##
#print ans.q.keywords
#print ans.q.searchwords