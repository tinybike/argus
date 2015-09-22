import requests
import json
from html_clean import sentence_split_guardian

def kw_to_query(keywords):
    query=''
    for word in keywords:
            query+=word+" AND "
    query=query[:-5]
    return query

def get_content(keywords):
    api_url = 'http://content.guardianapis.com/search'
    payload = {
        'q':                    kw_to_query(keywords),
        'from-date':            '2014-09-01',
        'to-date':              '2015-09-01',
        'api-key':              'qdz547b6gvss2ndwc9npwqcx',
        'page-size':            50,
        'format':               'json',
        'show-fields':          'all'

    }
    response = requests.get(api_url, params=payload)
    data = response.json()
    jobj=json.loads(json.dumps(data, indent=4))
#    print json.dumps(data, indent=4)
#    return search_headlines(keywords,jobj)
    return search_sentences(keywords,jobj)
    
    
def search_sentences(keywords,jobj):
    if len(jobj['response']['results'])==0:
        return (False,('absolutely no result','absolutely no result','absolutely no result'))
    for i in range(0,len(jobj['response']['results'])):
        bodyhtml=jobj['response']['results'][i]['fields']['body']
        sentences=sentence_split_guardian(bodyhtml)
#        print '\n-----\n'.join(sentences)
        
        for sentence in sentences:
            j=0
            for word in keywords:
                if word.lower() not in sentence.lower():
                    j+=1
                    break
            if j==0:
#                print 'found in:',sentence
                return (True,
                (jobj['response']['results'][i]['fields']['headline'], #headline
                jobj['response']['results'][i]['webUrl'],   #url
                jobj['response']['results'][i]['fields']['body']))  #body
    return (False,('no result','no result','no result'))

def search_headlines(keywords,jobj):
    if len(jobj['response']['results'])==0:
        return (False,('absolutely no result','absolutely no result','absolutely no result'))
    for i in range(0,len(jobj['response']['results'])):
        headline=jobj['response']['results'][i]['fields']['headline']
#        print headline
        j=0
        for word in keywords:
            if word not in headline:
                j+=1
                break
        if j==0:
            return (True,
            (jobj['response']['results'][i]['fields']['headline'], #headline
            jobj['response']['results'][i]['webUrl'],   #url
            jobj['response']['results'][i]['fields']['body']))  #body
    return (False,('no result','no result','no result'))
#get_content(['Barack','Obama','Nasa'])

