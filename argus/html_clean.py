# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import nltk.data
#from spacy.en import English
#nlp = English()
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#import time


def sentence_split(article):
#    startsp = time.time()
#    doc = nlp(article, entity=False)
#    endsp = time.time()
#    startnl = time.time()
#    tokenizer.tokenize(article)
#    endnl = time.time()
#
#    print '%.5f vs %.5f' % (endnl-startnl,endsp-startsp)
#    sentences = []
#    for sent in doc.sents:
#        sentences.append(sent.orth_)
#    return sentences
    return tokenizer.tokenize(article)

def preparse_guardian(html):
    html = html.replace(u"\\u201c", "\"")
    html = html.replace(u"\\u201d", "\"")
    html = html.replace(u"\\u2013", "-")
    html = html.replace(u"\\u2019", "'")
    html = html.replace(u"\\u2022", ".")
    html = html.replace(u"•", ".")
    soup = BeautifulSoup(html, "lxml")
    texts = soup.findAll(text=True)
    article = ''
    relatednext = False
    for block in texts:
        if relatednext or 'Related:' in block:
            relatednext = not relatednext
            continue
        article += block
    return article

def sentence_split_guardian(html):
    return sentence_split(preparse_guardian(html))
#sentence_split_guardian(html)
