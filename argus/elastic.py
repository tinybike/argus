# -*- coding: utf-8 -*-
"""Search for sentences containing searchwords
"""
from elasticsearch import Elasticsearch
from html_clean import sentence_split
from dateutil.parser import parse
import datetime
from answer import Source

es = Elasticsearch(hosts=['localhost', 'pasky.or.cz'])


def kw_to_query(keywords):
    query = ''
    for word in keywords:
        query += " " + word
    return query


def get_content_elastic(a, search_all=True):
    try:
        if len(a.q.date) > 0:
            d = a.q.date
            the_date = parse(d, ignoretz=True, fuzzy=True).date()
            from_date = the_date - datetime.timedelta(days=14)
            if the_date.day == 31 and the_date.month == 12:
                from_date = datetime.date(the_date.year, 1, 1)
            to_date = the_date + datetime.timedelta(days=14)
        else:
            from_date = datetime.date(2013, 1, 1)
            to_date = datetime.date(2016, 1, 1)
    except ValueError:
        print 'Caught ValueError: wrong date format of:', a.q.date_text
        pass
    q = {
        "query": {
            "filtered": {
                "query": {
                    "multi_match": {
                        "query": kw_to_query(a.q.searchwords),
                        "operator": "and",
                        "fields": ["headline^5", "summary^3", "body"]
                    }
                },
                "filter": {
                    "range": {"date": {"gte": from_date,
                                       "lte": to_date
                                       }}
                }
            }
        },
        #  "sort": { "date": { "order": "desc" }}
    }
    res = es.search(index="argus", size=100, body=q)
    return search_for_keywords(a, res, search_all)


def check_unknowns(a):
    for keyword in a.q.keywords:
        words = keyword.split()
        for word in words:
            q = {
                "query": {
                    "filtered": {
                        "query": {
                            "multi_match": {
                                "query": word,
                                "operator": "and",
                                "fields": ["headline^5", "summary^3", "body"]
                            }
                        }
                    }
                }
            }
            res = es.search(index="argus", size=100, body=q)
            if len(res['hits']['hits']) == 0:
                a.q.unknown.append(word)


def ask(a, query):
    q = {
        "query": {
            "filtered": {
                "query": {
                    "multi_match": {
                        "query": query,
                        "operator": "and",
                        "fields": ["headline^5", "summary^3", "body"]
                    }
                }
            }
        }
    }
    jobj = es.search(index="argus", size=100, body=q)
    if len(jobj['hits']['hits']) == 0:
        return False, False
    for i in range(len(jobj['hits']['hits'])):
        headline = jobj['hits']['hits'][i]['_source']['headline']
        summary = jobj['hits']['hits'][i]['_source']['summary']
        source = jobj['hits']['hits'][i]['_source']['source']
        url = jobj['hits']['hits'][i]['_source']['url']
        a.sources.append(Source(source, url, headline, summary, headline))
    if len(a.sources) != 0:
        return True, True
    return False, True


def search_for_keywords(a, jobj, search_all):
    if len(jobj['hits']['hits']) == 0:
        return False, False
    max_score = 1
    for i in range(0, len(jobj['hits']['hits'])):
        if max_score < float(jobj['hits']['hits'][i]['_score']):
            max_score = float(jobj['hits']['hits'][i]['_score'])
    max_score = 1
    for i in range(0, len(jobj['hits']['hits'])):
        headline = jobj['hits']['hits'][i]['_source']['headline']
        summary = jobj['hits']['hits'][i]['_source']['summary']
        source = jobj['hits']['hits'][i]['_source']['source']
        url = jobj['hits']['hits'][i]['_source']['url']
        date = jobj['hits']['hits'][i]['_source']['date']
        found, sentence = search_short(a, headline, search_all)
        if not found:
            found, sentence = search_sentences(a, summary, search_all)
            if not found:
                try:
                    body = jobj['hits']['hits'][i]['_source']['body']
                except KeyError:
                    continue
                found, sentence = search_sentences(a, body, search_all)
                if not found:
                    continue
        a.sources.append(Source(source, url, headline, summary,
                                sentence, date))
        a.sources[-1].elastic = float(jobj['hits']['hits'][i]['_score']) / max_score

    if len(a.sources) != 0:
        return True, True
    return False, True


def search_short(a, text, search_all=True):
    if not search_all:
        some = float(len(a.q.searchwords)) / 3 * 2  # change the fraction to search fore more/less kws
    else:
        some = len(a.q.searchwords)
    i = 0
    for word in a.q.searchwords:
        if word.lower() in text.lower():
            i += 1
    if i < some:
        return False, ''
    return True, text


def search_sentences(a, body, search_all):
    sentences = sentence_split(body)
    for sentence in sentences:
        found, text = search_short(a, sentence, search_all)
        if found:
            return True, text
    return False, ''
