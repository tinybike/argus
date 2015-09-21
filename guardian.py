import requests
import json
def get_content(query):
    api_url = 'http://content.guardianapis.com/search'
    payload = {
        'q':                    query,
        'from-date':            '2014-09-01',
        'to-date':              '2015-09-01',
        'api-key':              'qdz547b6gvss2ndwc9npwqcx',
        'page-size':            1,
        'format':               'json',
        'show-fields':          'all'

    }
    response = requests.get(api_url, params=payload)
    data = response.json()
    jobj=json.loads(json.dumps(data, indent=4))
#    print json.dumps(data, indent=4)
    if len(jobj['response']['results'])>0:
        return (True,
                (jobj['response']['results'][0]['fields']['headline'], #headline
                jobj['response']['results'][0]['webUrl'],   #url
                jobj['response']['results'][0]['fields']['body']))  #body
    else:
        return (False,('no result','no result','no result'))


#get_content('Barack AND Obama')

