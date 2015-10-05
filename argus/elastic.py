# -*- coding: utf-8 -*-
from elasticsearch import Elasticsearch
from html_clean import sentence_split
import sys
import json

#JSONFOLDER = 'sources/guardian_database'
es = Elasticsearch(hosts=['localhost', 'pasky.or.cz'])
#TODO: filter dates, search in headline+summary+body
def kw_to_query(keywords):
    query = ''
    for word in keywords:
            query += " "+word
    return query

def get_content_elastic(a):
    q = {"query":{
  "multi_match": {
    "query":    kw_to_query(a.q.searchwords),
    "operator": "and",
    "fields": [ "headline^5", "summary^3", "body" ]
  }}}
    res = es.search(index="test-index", size=100, body=q)
    return search_for_keywords(a, res)

def search_for_keywords(a,jobj):
    if len(jobj['hits']['hits']) == 0:
            return False, False
    for i in range(0, len(jobj['hits']['hits'])):
        headline = jobj['hits']['hits'][i]['_source']['headline']
        summary = jobj['hits']['hits'][i]['_source']['summary']
        source = jobj['hits']['hits'][i]['_source']['source']

        if not search_short(a, headline):
            if not search_short(a, summary):
                try:
                    body = jobj['hits']['hits'][i]['_source']['body']
                except KeyError:
                    continue
                if not search_body(a, body):
                    continue
        a.headlines.append(jobj['hits']['hits'][i]['_source']['headline'])
        a.urls.append(jobj['hits']['hits'][i]['_source']['url'])
        a.bodies.append(summary)
        a.sources.append(source)
    if len(a.urls) != 0:
        return True, True
    return False, True

def search_short(a,text):
    for word in a.q.keywords:
        if word.lower() not in text.lower():
            return False
    a.sentences.append(text)
    return True

def search_body(a, body):
    sentences = sentence_split(body)
    for sentence in sentences:
        if search_short(a, sentence):
            a.sentences.append(sentence)
            return True
    return False

def ask(query):
    q = {"query":{
  "multi_match": {
    "query":    query,
    "operator": "and",
    "fields": [ "headline^5", "summary^3", "body" ]
  }}}

    res = es.search(index="test-index", size=100, body=q)
    print("Got %d Hits:" % res['hits']['total'])
    for hit in res['hits']['hits']:
        try:
            print("%(headline)s " % hit["_source"])
        except KeyError:
            print('------------------------')
            continue
        print('------------------------')
    #
    with open('sources/elastictest.json', 'wb') as f:
        f.write(json.dumps(res, indent = 4))

