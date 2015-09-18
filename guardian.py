import requests
import json
#search?q=debate&tag=politics/politics&from-date=2014-01-01&api-key=test
def get_content(query):
    api_url = 'http://content.guardianapis.com/search'
    payload = {
        'q':                    query,
        'from-date':            '2014-09-01',
        'to-date':              '2015-09-01',
        'api-key':              'qdz547b6gvss2ndwc9npwqcx',
        'page-size':            1,
        'format':               'json',
        'show-fields':          'headline'

    }
    response = requests.get(api_url, params=payload)
    data = response.json()
    jobj=json.loads(json.dumps(data, indent=4))
    if len(jobj['response']['results'])>0:
        return (True,jobj['response']['results'][0]['fields']['headline'],)
    else:
        return (False,'no result')


#print get_content('was AND Barack AND Obama')

