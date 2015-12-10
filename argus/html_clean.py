# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
import re


def sentence_split(article):
    tokenized = tokenizer.tokenize(article)
    sentences = []
    for sentence in tokenized:
        sentences.append(sentence)
#        regex = re.match('^[A-Z]{2,}(.*)', sentence)
#        if regex:
#            sentences.append(regex.group(1))
#        else:
#            sentences.append(sentence)
    return sentences

def clean(html):
    soup = BeautifulSoup(html, "lxml")
    texts = soup.findAll(text=True)
    return ' '.join(texts)


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
