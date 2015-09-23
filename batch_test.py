import csv
import os
from main_frame import get_answer

#TSVFILE=sys.argv[1]
CSVFOLDER = "tests/batches"
OUTFILE = "tests/outfile.tsv"
#OUTFILE = "tests/outfile_zruseni didnt understand.tsv"
def reparse():
    qnum = 0

    with open(OUTFILE, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for csvfile in os.listdir(CSVFOLDER):
            i = 0
            for line in csv.reader(open(CSVFOLDER+'/'+csvfile), delimiter=',',skipinitialspace=True):
                if i == 0:
                    i += 1
                    info = ['HITID', 'Question', 'TurkAnswer', 'OurAnswer', 'OurKeywords', 'FoundSentence', 'OurHeadline', 'TurkTopic', 'TurkURL', 'OurURL']
                    if qnum == 0:
                        writer.writerow(info)
                    continue
                if line[16] == 'Rejected':
                    continue

                qnum += 1
                ouranswer = get_answer(line[30])
                (headline, url, body, sentence) = ouranswer.sources
                info=[line[0], line[30], line[28],
                      ouranswer.text, ouranswer.q.query, sentence, headline,line[31],line[29],url]
                info = [field.encode('utf-8') for field in info]
                writer.writerow(info)
                print 'answering question',qnum

def get_stats():
    i = -1
    correct = 0
    not_sure = 0
    yes = 0
    complete_query = 0
    answered = 0
    anr = 0
    understood = 0
    for line in csv.reader(open(OUTFILE), delimiter='\t'):
        i += 1
        if i == 0:
            continue
        turkans = line[2]
        ourans = line[3]
        if ourans == 'Didn\'t understand the question':
            continue
        understood += 1
        if ourans in 'Not sure':
            not_sure+=1
        if turkans == ourans:
            correct += 1
        if turkans == 'YES':
            yes += 1
        if '(' not in line[4]:
            complete_query += 1
        if ourans in 'YES NO':
            answered += 1
        if line[5] == 'absolutely no result':
            anr += 1

    precision = float(correct) / float(answered)
    recall = float(correct)/float(understood)
    print 'Answered = ', answered
    print 'Recall =', recall
    print 'Precision =', precision
    print 'not sure:', not_sure, '/', understood
    print 'complete query:', complete_query, '/', i
    print 'Absolutely no result in', anr, '/', understood


def turkstats():
    i = -1
    sport = 0
    stock = 0
    politics = 0
    for line in csv.reader(open(OUTFILE), delimiter='\t'):
        i += 1
        if i == 0:
            continue
        if line[7] == 'sport':
            sport += 1
        if line[7] == 'stock market':
            stock += 1
        if line[7] == 'politics':
            politics += 1
    print 'sport',float(sport)/i
    print 'politics',float(politics)/i
    print 'stock market',float(stock)/i

if __name__ == "__main__":
    reparse()
    get_stats()
#    turkstats()