import csv
from main_frame import get_answer,get_sources,get_query



#TSVFILE=sys.argv[1]
TSVFILE="testbatch/Batch_2094072_batch_results_cleaned.csv"
OUTFILE="testbatch/outfile.csv"
def reparse():
    i=0
    with open(OUTFILE, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        for line in csv.reader(open(TSVFILE), delimiter=',',skipinitialspace=True):
            if i==0:
#                labels=line
                i+=1
                info=['HITID','Question','TurkAnswer','OurAnswer','OurKeywords','OurHeadline','TurkTopic','TurkURL','OurURL']
                writer.writerow(info)
                continue
            if line[16]=='Rejected':
                continue
            
            ouranswer=get_answer(line[30])
            (headline,url,body)=get_sources()
            info=[line[0],line[30],line[28],ouranswer,get_query(),headline,line[31],line[29],url]
            writer.writerow(info)
    #        print "HITID=",line[0],", Question=",line[30],", Answer=",line[28]

reparse()

def get_stats():
    i=-1
    perc=0
    for line in csv.reader(open(OUTFILE), delimiter='\t'):
        i+=1
        if i==0:
            continue
        tans='YES'
        if line[2]=='0':
            tans='NO'
        if tans==line[3]:
            perc+=1
    print 'Total questions:',i,', %correct:',float(perc)/float(i)*100

get_stats()