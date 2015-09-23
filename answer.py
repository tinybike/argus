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


class Question(object):
    text = ''
    keywords = []
    postokens = []
    query = ''
    def __init__(self,question):
        self.text = question
        self.keywords = extract(self)
        self.query = kw_to_query(self.keywords)



#ans = Answer(Question('Will the Giants win the World Series in 2014?'))
#
#
#print ans.q.keywords