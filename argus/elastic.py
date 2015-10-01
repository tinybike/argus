# -*- coding: utf-8 -*-
from datetime import datetime
from elasticsearch import Elasticsearch
from html_clean import preparse_guardian, sentence_split
import sys
import json
import os

#JSONFOLDER = 'sources/guardian_database'
es = Elasticsearch(hosts=['localhost', 'pasky.or.cz'])
#TODO: filter dates, search in headline+summary+body
def fill_guardian(JSONFOLDER):
    ID = 0
    for jsonfile in os.listdir(JSONFOLDER):
        if not jsonfile.endswith(".json"):
            continue
        with open(JSONFOLDER+'/'+jsonfile) as data_file:
            jobj = json.load(data_file)
        for i in range(0, len(jobj['response']['results'])):
            try:
                if jobj['response']['results'][i]['type'] != 'article':
                    continue
                headline = jobj['response']['results'][i]['fields']['headline']
                date = datetime.strptime(jobj['response']['results'][i]['webPublicationDate'], "%Y-%m-%dT%H:%M:%SZ").date()
                url = jobj['response']['results'][i]['webUrl']
                bodyhtml = jobj['response']['results'][i]['fields']['body']
                source = jsonfile.split('_')[0]
                summaryhtml = jobj['response']['results'][i]['fields']['standfirst']
            except KeyError:
                continue
            body = preparse_guardian(bodyhtml)
            summary = preparse_guardian(summaryhtml)
            doc = {
                'headline':         headline,
                'date':             date,
                'url':              url,
                'body':             body,
                'source':           source,
                'summary':          summary,
            }
            ID += 1
            es.index(index="test-index", doc_type='article', body=doc)
            if ID % 100 == 0:
                print 'added article number', ID

    es.indices.refresh(index="test-index")
    return "ok"


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
    return search_sentences(a, res)

def search_sentences(a, jobj):
    try:
        if len(jobj['hits']['hits']) == 0:
            return False, False
    except KeyError:
        print 'Unknown error occured while answering:',a.q.text
        return False, False
    for i in range(0, len(jobj['hits']['hits'])):
        try:
            body = jobj['hits']['hits'][i]['_source']['body']
        except KeyError:
            continue

        sentences = sentence_split(body)

        for sentence in sentences:
            j = 0
            for word in a.q.keywords:
                if word.lower() not in sentence.lower():
                    j += 1
                    break
            if j == 0:
                a.headlines.append(jobj['hits']['hits'][i]['_source']['headline'])
                a.urls.append(jobj['hits']['hits'][i]['_source']['url'])
                a.bodies.append(body)
                a.sentences.append(sentence)
                a.sources.append('guardian')
    if len(a.urls) != 0:
        return True, True
    return False, True

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


if __name__ == "__main__":
    JSONFOLDER = sys.argv[1]
    fill_guardian(JSONFOLDER)
#    ask('Series Yankees')
