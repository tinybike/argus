# -*- coding: utf-8 -*-
from datetime import datetime
from elasticsearch import Elasticsearch
from argus.html_clean import preparse_guardian
import sys
import json
import os

JSONFOLDER = 'sources/nytimes_database'
es = Elasticsearch()
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


def fill_nytimes(JSONFOLDER):
    ID = 0
    for jsonfile in os.listdir(JSONFOLDER):
        if not jsonfile.endswith(".json"):
            continue
        with open(JSONFOLDER+'/'+jsonfile) as data_file:
            jobj = json.load(data_file)
        for i in range(0, len(jobj['response']['docs'])):
            try:
#                if jobj['response']['results'][i]['type'] != 'article':
#                    continue
                headline = jobj['response']['docs'][i]['headline']['main']
                date = datetime.strptime(jobj['response']['docs'][i]['pub_date'], "%Y-%m-%dT%H:%M:%SZ").date()
                url = jobj['response']['docs'][i]['web_url']
                source = jsonfile.split('_')[0]
                summary = jobj['response']['docs'][i]['abstract']
                if summary == None:
                    summary = jobj['response']['docs'][i]['lead_paragraph']
            except KeyError:
                continue
            doc = {
                'headline':         headline,
                'date':             date,
                'url':              url,
                'source':           source,
                'summary':          summary,
            }
            ID += 1
            es.index(index="test-index", doc_type='article', body=doc)
            if ID % 100 == 0:
                print 'added article number', ID

    es.indices.refresh(index="test-index")
    return "ok"

def ask(query):
    q = {
  "query": {
    "filtered": {
      "query": {
        "multi_match": {
            "query":    query,
            "operator": "and",
            "fields": [ "headline^5", "summary^3", "body" ]
            }
      },
      "filter": {
        "range": { "date": { "gte": "2015-06-14",
                             "lte": "2015-08-01"}}
      }
    }
  }
}
    res = es.search(index="test-index", size=100, body=q)
    print("Got %d Hits:" % res['hits']['total'])
    for hit in res['hits']['hits']:
        try:
            print("%(headline)s %(date)s" % hit["_source"])
        except KeyError:
            print('------------------------')
            continue
        print('------------------------')
    #
    with open('sources/elastictest.json', 'wb') as f:
        f.write(json.dumps(res, indent = 4))


if __name__ == "__main__":
    nyf = ''
    gf = ''
    for i in range(0,len(sys.argv)):
        if sys.argv[i][:3] == '-NY':
            nyf = sys.argv[i][3:]
        if sys.argv[i][:2] == '-G':
            gf = sys.argv[i][2:]
    if len(nyf) != 0:
        fill_nytimes(nyf)
    if len(gf) != 0:
        fill_guardian(gf)
#    ask('Saina Nehwal')
