# -*- coding: utf-8 -*-
from elasticsearch import Elasticsearch
from html_clean import sentence_split
from dateutil.parser import parse
import datetime


#JSONFOLDER = 'sources/guardian_database'
es = Elasticsearch(hosts=['localhost']) #, pasky.or.cz
def kw_to_query(keywords):
    query = ''
    for word in keywords:
            query += " "+word
    return query

def get_content_elastic(a):
    the_date = "2015-10-01"
    try:
        if len(a.q.date) > 0:
            d = a.q.date
            the_date = parse(d, ignoretz=True, fuzzy=True).date()
            from_date = the_date - datetime.timedelta(days=14)
            if the_date.day == 31 and the_date.month == 12:
                from_date = datetime.date(the_date.year, 1, 1)
            to_date = the_date + datetime.timedelta(days=14)
        else:
            from_date = datetime.date(2013,1,1)
            to_date = datetime.date(2016,1,1)
    except ValueError:
        print 'Caught ValueError: wrong date format of:',a.q.date_text
        pass
#    print a.q.date_text,'----->',to_date
    q = {
  "query": {
    "filtered": {
      "query": {
        "multi_match": {
            "query":    kw_to_query(a.q.searchwords),
            "operator": "and",
            "fields": [ "headline^5", "summary^3", "body" ]
            }
      },
      "filter": {
        "range": { "date": { "gte": from_date,
                             "lte": to_date
                             }}
      }
    }
  },
#  "sort": { "date": { "order": "desc" }}
}
    res = es.search(index="argus", size=100, body=q)
    return search_for_keywords(a, res)

def search_for_keywords(a,jobj):
    if len(jobj['hits']['hits']) == 0:
            return False, False
    for i in range(0, len(jobj['hits']['hits'])):
        headline = jobj['hits']['hits'][i]['_source']['headline']
        summary = jobj['hits']['hits'][i]['_source']['summary']
        source = jobj['hits']['hits'][i]['_source']['source']

        if not search_short(a, headline):
            if not search_sentences(a, summary):
                try:
                    body = jobj['hits']['hits'][i]['_source']['body']
                except KeyError:
                    continue
                if not search_sentences(a, body):
                    continue
        a.headlines.append(jobj['hits']['hits'][i]['_source']['headline'])
        a.urls.append(jobj['hits']['hits'][i]['_source']['url'])
        a.bodies.append(summary)
        a.sources.append(source)
        a.elastic.append(float(jobj['hits']['hits'][i]['_score']))
    if len(a.urls) != 0:
        return True, True
    return False, True

#XXX:appends sentence
def search_short(a,text):

    for word in a.q.searchwords:
        if word.lower() not in text.lower():
            return False
    a.sentences.append(text)
    return True

def search_sentences(a, body):
    sentences = sentence_split(body)
    for sentence in sentences:
        if search_short(a, sentence):
            return True
    return False

def check_unknowns(a):
    for keyword in a.q.keywords:
        words = keyword.split()
        for word in words:
            q = {
  "query": {
    "filtered": {
      "query": {
        "multi_match": {
            "query":    word,
            "operator": "and",
            "fields": [ "headline^5", "summary^3", "body" ]
            }
      }
    }
  }
}
            res = es.search(index="argus", size=100, body=q)
#            print 'searching for',word,', found:', len(res['hits']['hits'])
            if len(res['hits']['hits']) == 0:
                a.q.unknown.append(word)

def ask(a,query):
    q = {
  "query": {
    "filtered": {
      "query": {
        "multi_match": {
            "query":    query,
            "operator": "and",
            "fields": [ "headline^5", "summary^3", "body" ]
            }
      }
    }
  }
}
    jobj = es.search(index="argus", size=100, body=q)
    if len(jobj['hits']['hits']) == 0:
            return False, False
    for i in range(0, len(jobj['hits']['hits'])):
        headline = jobj['hits']['hits'][i]['_source']['headline']
        summary = jobj['hits']['hits'][i]['_source']['summary']
        source = jobj['hits']['hits'][i]['_source']['source']
        a.headlines.append(headline)
        a.sentences.append(headline)
        a.urls.append(jobj['hits']['hits'][i]['_source']['url'])
        a.bodies.append(summary)
        a.sources.append(source)
    if len(a.urls) != 0:
        return True, True
    return False, True
