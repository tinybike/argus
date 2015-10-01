import requests
import json

def get_content(page):
    api_url = 'http://api.nytimes.com/svc/search/v2/articlesearch.json'
    payload = {
        'begin_date':           '20140901',
        'end_date':             '20150901',
        'api-key':              'da3f5d9d42b3a9d28f7bc2951909f167:15:73058897',
        'sort':                 'oldest',
        'page':                 page,
    }
    response = requests.get(api_url, params=payload)
    data = response.json()
#    print json.dumps(data, indent=4)
    return json.dumps(data, indent = 4)

def download():
    hits = int(json.loads(get_content(0))['response']['meta']['hits'])
    p = int(hits/10)+1
    for page in range(500,10000):
        with open('sources/nytimes_database/nytimes_'+str(page)+'.json', 'wb') as f:
            f.write(get_content(page))
        print 'reading page %d/%d' % (page,p)

if __name__ == '__main__':
    download()
