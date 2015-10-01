import requests
import json
from html_clean import sentence_split

def kw_to_query(keywords):
    query = ''
    for word in keywords:
            query += word + " AND "
    query = query[:-5]
    return query

def get_content_nytimes(a):
    api_url = 'http://api.nytimes.com/svc/search/v2/articlesearch.json'
    payload = {
        'q':                    kw_to_query(a.q.keywords),
        'begin_date':           '20140901',
        'end_date':             '20150901',
        'api-key':              'da3f5d9d42b3a9d28f7bc2951909f167:15:73058897',
        'sort':                 'newest',
#        'orderBy':              'newest',
    }
    response = requests.get(api_url, params=payload)
    data = response.json()
    jobj=json.loads(json.dumps(data, indent = 4))
#    print kw_to_query(a.q.searchwords)
#    print json.dumps(data, indent=4)
    return search_sentences(a, jobj)

def search_sentences(a, jobj):
    try:
        if len(jobj['response']['docs']) == 0:
            return False, False
    except KeyError:
        print 'Unknown error occured while answering:',a.q.text
        return False, False

    for i in range(0, len(jobj['response']['docs'])):
        try:
            body = jobj['response']['docs'][i]['lead_paragraph']
            if body == None:
                body = jobj['response']['docs'][i]['abstract']
#            print body
        except KeyError:
            continue

        sentences = sentence_split(body)
#        print '\n-----\n'.join(sentences)

        for sentence in sentences:
            j = 0
            for word in a.q.keywords:
                if word.lower() not in sentence.lower():
                    j += 1
                    break
            if j == 0:
                a.headlines.append(jobj['response']['docs'][i]['headline']['main'])
                a.urls.append(jobj['response']['docs'][i]['web_url'])
                a.bodies.append(body)
                a.sentences.append(sentence)
                a.sources.append('nytimes')

    if len(a.urls) != 0:
        return True, True
    return False, True
