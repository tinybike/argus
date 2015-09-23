import csv
import os
from main_frame import get_answer, get_sources, get_query

#TSVFILE=sys.argv[1]
CSVFOLDER = "tests/batches"
#OUTFILE = "tests/outfile.tsv"
OUTFILE = "tests/outfile_zruseni didnt understand.tsv"
def reparse():
    qnum = 0

    with open(OUTFILE, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for csvfile in os.listdir(CSVFOLDER):
            i = 0
            for line in csv.reader(open(CSVFOLDER+'/'+csvfile), delimiter=',',skipinitialspace=True):
                if i == 0:
    #                labels=line
                    i += 1
                    info = ['HITID', 'Question', 'TurkAnswer', 'OurAnswer', 'OurKeywords', 'OurHeadline', 'TurkTopic', 'TurkURL', 'OurURL']
                    if qnum == 0:
                        writer.writerow(info)
                    continue
                if line[16] == 'Rejected':
                    continue
                qnum += 1
                ouranswer = get_answer(line[30])
                (headline, url, body) = get_sources()
                info=[line[0].encode('utf-8'), line[30].encode('utf-8'), line[28].encode('utf-8'),
                      ouranswer.encode('utf-8'), get_query().encode('utf-8'), headline.encode('utf-8'),line[31].encode('utf-8'),line[29].encode('utf-8'),url.encode('utf-8')]

                writer.writerow(info)
#                print 'answering question:', line[30],'(',qnum,')'
                print 'answering question',qnum

def get_stats():
    i = -1
    correct = 0
    not_sure = 0
    yes = 0
    complete_query = 0
    answered = 0
    anr = 0
    for line in csv.reader(open(OUTFILE), delimiter='\t'):
        i += 1
        if i == 0:
            continue

        if line[3] in 'Not sure':
            not_sure+=1
        if line[2] == line[3]:
            correct += 1

        if line[2] == 'YES':
            yes += 1

        if '(' not in line[4]:
            complete_query += 1
        if line[3] in 'YES NO':
            answered+=1

        if line[5] == 'absolutely no result':
            anr+=1

    precision = float(correct) / float(answered)
    recall = float(correct)/float(i)
    print 'Answered = ',answered
    print 'Recall =', recall
    print 'Precision =', precision
#    print 'YES=', float(yes)/float(i)*100, '%'
    print 'not sure:', not_sure, '/', i
    print 'complete query:', complete_query, '/', i
    print 'Absolutely no result in',anr, '/', i

if __name__ == "__main__":
#    reparse()
    get_stats()