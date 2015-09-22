import csv
import os
from main_frame import get_answer,get_sources,get_query



#TSVFILE=sys.argv[1]
CSVFOLDER="tests/batches"
OUTFILE="tests/outfile.csv"
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
                info=[line[0],line[30],line[28],ouranswer,get_query(),headline,line[31],line[29],url]
                writer.writerow(info)
                print 'question number',qnum

def get_stats():
    i=-1
    perc=0
    false_positive=0
    false_negative=0
    not_sure=0
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
            not_sure+=1
        
    print 'Total questions:',i,',',float(perc)/float(i)*100,'% correct:'
    print 'not sure:',not_sure,'/',i
    print 'false positive:',false_positive,'/',i
    print 'false negative:',false_negative,'/',i

#reparse()
get_stats()