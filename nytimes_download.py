import requests
import json
import math

months = ['20140901', '20140911', '20140912', '20140921', '20140922', '20141001', '20141002', '20141011', '20141012', '20141021', '20141022', '20141101', '20141102', '20141111', '20141112', '20141121', '20141122', '20141201', '20141202', '20141211', '20141212', '20141221', '20141222', '20150101', '20150102', '20150111', '20150112', '20150121', '20150122', '20150201', '20150202', '20150211', '20150212', '20150221', '20150222', '20150301', '20150302', '20150311', '20150312', '20150321', '20150322', '20150401', '20150402', '20150411', '20150412', '20150421', '20150422', '20150501', '20150502', '20150511', '20150512', '20150521', '20150522', '20150601', '20150602', '20150611', '20150612', '20150621', '20150622', '20150701', '20150702', '20150711', '20150712', '20150721', '20150722', '20150801', '20150802', '20150811', '20150812', '20150821', '20150822', '20150901']

def get_content(page,begin_date,end_date):
    api_url = 'http://api.nytimes.com/svc/search/v2/articlesearch.json'
    payload = {
        'fq':                   'news_desk:("Sports" "Politics" "Business")',
        'begin_date':           begin_date,
        'end_date':             end_date,
        'api-key':              'da3f5d9d42b3a9d28f7bc2951909f167:15:73058897',
        'sort':                 'oldest',
        'page':                 page,
    }
    response = requests.get(api_url, params=payload)
    data = response.json()
#    print json.dumps(data, indent=4)
    return json.dumps(data, indent = 4)

def download():
    hits = int(json.loads(get_content(0,'20140901','20150901'))['response']['meta']['hits'])
    allp = int(math.ceil(hits/10))
    currp = 0
    for i in xrange(0,len(months),2):
        begin_date = months[i]
        end_date = months[i+1]
        hits = int(json.loads(get_content(0,begin_date,end_date))['response']['meta']['hits'])
        p = int(math.ceil(hits/10))
        print "%d pages from %s to %s" % (p,begin_date,end_date)
        for page in range(0,p+1):
            with open('sources/nytimes_database/nytimes_'+begin_date+'_'+end_date+'_'+str(page)+'.json', 'wb') as f:
                f.write(get_content(page,begin_date,end_date))
            currp += 1
            print 'reading page %d/%d (%d/%d)' % (page,p,currp,allp)

if __name__ == '__main__':
    download()




#for year in range(2014,2016):
#    for month in range(1,13):
#        if (year == 2014 and month < 9) or (year == 2015 and month > 9):
#            continue;
#        print "\'%d%.2d01\'," % (year,month),
#        print "\'%d%.2d02\'," % (year,month),
#        print "\'%d%.2d11\'," % (year,month),
#        print "\'%d%.2d12\'," % (year,month),
#        print "\'%d%.2d21\'," % (year,month),
#        print "\'%d%.2d22\'," % (year,month),
