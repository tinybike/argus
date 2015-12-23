import requests
import json


def get_content(page):
    api_url = 'http://content.guardianapis.com/search'
    payload = {
        'from-date': '2014-09-01',
        'to-date': '2015-09-01',
        'api-key': 'qdz547b6gvss2ndwc9npwqcx',
        'page-size': 50,
        'page': page,
        'format': 'json',
        'orderBy': 'newest',
        'show-fields': 'all'
    }
    response = requests.get(api_url, params=payload)
    data = response.json()
    return json.dumps(data, indent=4)


def download():
    p = int(json.loads(get_content(1))['response']['pages'])
    for page in range(1, p):
        with open('sources/database/guardian_' + str(page) + '.json', 'wb') as f:
            f.write(get_content(page))
        print 'reading page', page


if __name__ == '__main__':
    download()
