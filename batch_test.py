import csv
import os
from main_frame import get_answer, get_sources, get_query

#TSVFILE=sys.argv[1]
CSVFOLDER="tests/batches"
OUTFILE="tests/outfile.tsv"
def reparse():
    qnum=0
    
    with open(OUTFILE, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for csvfile in os.listdir(CSVFOLDER):
            i=0
            for line in csv.reader(open(CSVFOLDER+'/'+csvfile), delimiter=',',skipinitialspace=True):
                if i==0:
    #                labels=line
                    i+=1
                    info=['HITID','Question','TurkAnswer','OurAnswer','OurKeywords','OurHeadline','TurkTopic','TurkURL','OurURL']
                    if qnum==0:
                        writer.writerow(info)
                    continue
                if line[16]=='Rejected':
                    continue
                qnum+=1
                ouranswer=get_answer(line[30])
                (headline,url,body)=get_sources()
                info=[line[0].encode('utf-8'),line[30].encode('utf-8'),line[28].encode('utf-8'),
                      ouranswer.encode('utf-8'),get_query().encode('utf-8'),headline.encode('utf-8'),line[31].encode('utf-8'),line[29].encode('utf-8'),url.encode('utf-8')]
                
                writer.writerow(info)
                print 'answering question number',qnum

def get_stats():
    i=-1
    perc=0
    false_positive=0
    false_negative=0
    not_sure=0
    no=0
    yes=0
    complete_query=0
    for line in csv.reader(open(OUTFILE), delimiter='\t'):
        i+=1
        if i==0:
            continue
        if line[2]==line[3]:
            perc+=1
        elif line[3]=='YES':
            false_positive+=1
        elif line[3]=='NO':
            false_negative+=1
        else:
#            perc+=1
            not_sure+=1
        if line[2]=='YES':
            yes+=1
        else:
            no+=1
        if '(' not in line[4]:
            complete_query+=1
        
    print 'Total questions:',i,',',float(perc)/float(i)*100,'% correct:'
    print 'YES=',float(yes)/float(i)*100,'%'
    print 'false positive:',false_positive,'/',i
    print 'false negative:',false_negative,'/',i
    print 'not sure:',not_sure,'/',i
    print 'complete query:',complete_query,'/',i
reparse()
get_stats()