# -*- coding: utf-8 -*-
from datetime import datetime
from elasticsearch import Elasticsearch
from argus.html_clean import preparse_guardian, clean
from dateutil.parser import parse
import feedparser
import hashlib
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
            uniqueid = hashlib.md5((headline+summary).encode('utf-8')).hexdigest()
            ID += 1
            es.index(index="test-index", doc_type='article', body=doc, id=uniqueid)
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
            uniqueid = hashlib.md5((headline+summary).encode('utf-8')).hexdigest()
            ID += 1
            es.index(index="test-index", doc_type='article', body=doc, id=uniqueid)
            if ID % 100 == 0:
                print 'added article number', ID

    es.indices.refresh(index="test-index")
    return "ok"


def fill_rss(RSSFOLDER):
    ID = 0
    for root, dirs, files in os.walk(RSSFOLDER):
        for name in files:
#            if not name.endswith(('.rss', '.xml')):
#                continue
            d = feedparser.parse(os.path.join(root, name))
            for entry in d.entries:
                headline = entry.title
                date = parse(entry.published).date()
                url = entry.link
                source = 'RSS'
                summary = clean(entry.description)
                if len(summary) == 0 or summary.isspace():
                    continue
                doc = {
                    'headline':         headline,
                    'date':             date,
                    'url':              url,
                    'source':           source,
                    'summary':          summary,
                }
                uniqueid = hashlib.md5((headline+summary).encode('utf-8')).hexdigest()
                ID += 1
                es.index(index="test-index", doc_type='article', body=doc, id=uniqueid)
                if ID % 100 == 0:
                    print 'added article number', ID

    es.indices.refresh(index="test-index")
    return "ok"

if __name__ == "__main__":
    nyf = ''
    gf = ''
    rssf = ''
    for i in range(len(sys.argv)):
        print sys.argv[i]
        if sys.argv[i][:3] == '-NY':
            nyf = sys.argv[i][3:]
        if sys.argv[i][:2] == '-G':
            gf = sys.argv[i][2:]
        if sys.argv[i][:4] == '-RSS':
            rssf = sys.argv[i][4:]
    if len(nyf) != 0:
        fill_nytimes(nyf)
    if len(gf) != 0:
        fill_guardian(gf)
    if len(rssf) != 0:
        fill_rss(rssf)
