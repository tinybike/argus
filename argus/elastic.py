# -*- coding: utf-8 -*-
"""Search for sentences containing searchwords
"""

import datetime
from dateutil.parser import parse
from elasticsearch import Elasticsearch
from html_clean import sentence_split

from answer import Source

es = Elasticsearch(hosts=['localhost', 'pasky.or.cz'])


def get_content_elastic(a, search_all=True):
    from_date_ = datetime.date(2013, 1, 1)
    to_date_ = datetime.date(2016, 1, 1)

    from_date, to_date, is_sloped = a.q.date_period()
    if from_date is None:
        return get_content_elastic_q(a, a.q.searchwords, from_date_, to_date_, search_all=search_all)

    # If date is set, use a hybrid strategy - query with the date
    # *either* as time restriction or as a fulltext searchword.
    # We can't just do the former as questions like "run for 2016
    # presidency" may appear.
    fs0, fa0 = get_content_elastic_q(a, a.q.searchwords, from_date, to_date, search_all=search_all)
    fs1, fa1 = get_content_elastic_q(a, list(a.q.searchwords) + [a.q.date_texts[0]], from_date_, to_date_, search_all=search_all)

    return ((fs0 or fs1), (fa0 or fa1))


def get_content_elastic_q(a, ft_searchwords, from_date, to_date, search_all=True):
    """ fill the Answer object with sources based on elasticsearch query,
    from the given date period """
    q = {
        "query": {
            "filtered": {
                "query": {
                    "multi_match": {
                        "query": ' '.join(ft_searchwords),
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
    for keyword in a.q.all_keywords():
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
        date = parse(jobj['hits']['hits'][i]['_source']['date'], ignoretz=True, fuzzy=True).date()
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
        some = float(len(a.q.searchwords)) * 2/3  # change the fraction to search fore more/less kws
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
