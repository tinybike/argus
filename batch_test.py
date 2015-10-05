from __future__ import division
import csv
import os
from argus.main_frame import get_answer

CSVFOLDER = "tests/batches"
OUTFILE = "tests/outfile.tsv"
def reparse():
    qnum = 0
    with open(OUTFILE, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        for csvfile in os.listdir(CSVFOLDER):
            if not csvfile.endswith(".csv"):
                continue
            i = 0
            for line in csv.reader(open(CSVFOLDER+'/'+csvfile), delimiter=',',skipinitialspace=True):
                if i == 0:
                    i += 1
                    info = ['HITID', 'Question', 'TurkAnswer', 'OurAnswer',
                            'OurKeywords', 'FoundSentence', 'OurHeadline',
                            'TurkTopic', 'TurkURL', 'OurURL','Sentiment','Source', 'info']
                    if qnum == 0:
                        writer.writerow(info)
                    continue
                if line[16] == 'Rejected':
                    continue
                qnum += 1
                ouranswer = get_answer(line[30])

                url = ''
                headline = ''
                sentence = ''
                source = ''
                sentiment = ''
                if len(ouranswer.urls) != 0:
                    url = ouranswer.urls[0]
                    headline = ouranswer.headlines[0]
                    sentence = ouranswer.sentences[0]
                    source = ouranswer.sources[0]
                    for j in range(0,len(ouranswer.sentiment)):
                        sentiment += str(ouranswer.sentiment[j][0])+" "+str(ouranswer.sentiment[j][1])+" "+str(ouranswer.sentiment[j][2])+":"
                    sentiment = sentiment[:-1]
                info = [line[0], line[30], line[28],
                      ouranswer.text, ouranswer.q.query, sentence,
                      headline,line[31],line[29],url,sentiment,source,ouranswer.info]
                info = [field.encode('utf-8') for field in info]
                writer.writerow(info)
                if qnum % 10 == 0:
                    print 'answering question',qnum

def get_stats():
    i = -1
    correct = 0
    not_sure = 0
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
        if ourans == 'Not sure':
            not_sure+=1
        if turkans == ourans:
            correct += 1
        if ourans in 'YES NO':
            answered += 1
        if ourans == 'Absolutely not sure':
            anr += 1

    precision = correct / answered
    recall = correct / understood
    print 'Out of %d questions we understand %d (%.2f%%)' % (i, understood, understood/i*100)
    print 'Out of these %d questions:' % understood
    print 'We didnt find any articles containing all searchwords in %d (%.2f%%) cases' % (anr, anr/understood*100)
    print 'We didnt find any sentences containing all keywords in %d (%.2f%%) cases' % (not_sure, not_sure/understood*100)
    print 'We were able to answer %d-%d-%d = %d (%.2f%%) questions' % (understood, anr, not_sure, answered, answered/understood*100)
    print 'Recall =', recall
    print 'Precision =', precision

def parse_yes():
    i=0
    with open('tests/erroranalysis.tsv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        for line in csv.reader(open(OUTFILE), delimiter='\t'):
            if (line [2] in 'NO YES' and line[3] in 'YES NO') or i==0:
                writer.writerow(line)
                i += 1

def turkstats():
    i = -1
    sport = 0
    stock = 0
    politics = 0
    yes = 0
    for line in csv.reader(open(OUTFILE), delimiter='\t'):
        i += 1
        if i == 0:
            continue
        if line[2] == 'YES':
            yes += 1
        if line[7] == 'sport':
            sport += 1
        if line[7] == 'stock market':
            stock += 1
        if line[7] == 'politics':
            politics += 1
    print '\n----------\n'
    print 'YES answered in %.2f%% of turk answers' % (yes/i*100)
    print 'Topics:'
    print 'sport %.2f%%' % (sport/i*100)
    print 'politics %.2f%%' % (politics/i*100)
    print 'stock market %.2f%%' % (stock/i*100)

if __name__ == "__main__":
    reparse()
    get_stats()
    turkstats()
#    parse_yes()
