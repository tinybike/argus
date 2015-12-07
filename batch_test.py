from __future__ import division
import csv
import os
from argus.main_frame import get_answer
import numpy as np
from argus.features import feature_list_official


CSVFOLDER = "tests/batches"
trainIDs = np.load('tests/trainIDs/trainIDs.npy')

def reparse():
    qnum = 0
    info_file = open(INFOFILE, 'wb')
    info_writer = csv.writer(info_file, delimiter='\t')
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
                            'TurkTopic', 'TurkURL', 'OurURL','Source', 'info']
                    info += feature_list_official
                    if qnum == 0:
                        writer.writerow(info)
                        info_writer.writerow(['Question', 'Sentence','verbs','wn','spacy'])
                    continue
                if line[16] == 'Rejected':
                    continue
                qnum += 1
                ouranswer = get_answer(line[30])

                for source in ouranswer.sources:
                    info = []
                    info.append(ouranswer.q.text)
                    info.append(source.sentence)
                    info.append(source.features[5].get_info())
                    info.append(str(source.features[5].get_value()))
                    info.append(str(source.features[4].get_value()))
                    info = [field.encode('utf-8') for field in info]
                    info_writer.writerow(info)


                url = ''
                headline = ''
                sentence = ''
                source = ''
                feat = ''
                info = []
                if len(ouranswer.sources) != 0:
                    url = ouranswer.sources[0].url
                    headline = ouranswer.sources[0].headline
                    sentence = ouranswer.sources[0].sentence
                    source = ouranswer.sources[0].source
                    for j in range(len(feature_list_official)):
                        for s in ouranswer.sources:
                            feat += str(s.features[j].get_value())+":"
                        feat = feat[:-1]
                        info.append(feat)
                        feat = ''
                info = [line[0], line[30], line[28],
                        ouranswer.text, ouranswer.q.query, sentence,
                        headline, line[31], line[29], url, source,
                        ouranswer.info]+info
                info = [field.encode('utf-8') for field in info]
                writer.writerow(info)
                if qnum % 10 == 0:
                    print 'answering question', qnum
    info_file.close()

def get_stats():
    i = -1
    correct = 0
    no_result = 0
    answered = 0
    anr = 0
    understood = 0
    trainedon = 0
    yes = 0
    for line in csv.reader(open(OUTFILE), delimiter='\t'):
        i += 1
        if i == 0:
            continue
        if validation:
            if line[1] in trainIDs:
                trainedon += 1
                i-=1
                continue
        turkans = line[2]
        ourans = line[3]
        if ourans == 'Didn\'t understand the question':
            continue
        understood += 1
        if ourans in 'YES NO':
            answered += 1
            if turkans == 'YES':
                yes +=1
        if ourans == 'No result':
            no_result += 1
        if turkans == ourans:
            correct += 1
        if ourans == 'Absolutely no result':
            anr += 1


    precision = correct / answered
    recall = correct / understood
    print 'Out of %d questions we understand %d (%.2f%%)' % (i, understood, understood/i*100)
    print 'Out of these %d questions:' % understood
    print 'We didnt find any articles containing all searchwords in %d (%.2f%%) cases' % (anr, anr/understood*100)
    print 'We didnt find any sentences containing all keywords in %d (%.2f%%) cases' % (no_result, no_result/understood*100)
    print 'We were able to answer %d-%d-%d = %d (%.2f%%) questions' % (understood, anr, no_result, answered, answered/understood*100)
    print 'Recall =', recall
    print 'Precision =', precision
    print 'Turk answered YES in %.2f%% of answered' % (yes/answered*100)

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
    print 'YES answered in %.2f%% of turk answers' % (yes/i*100)
    print 'Topics:'
    print 'sport %.2f%%' % (sport/i*100)
    print 'politics %.2f%%' % (politics/i*100)
    print 'stock market %.2f%%' % (stock/i*100)


def more_stats():
    i = 0
    y = 0
    for line in csv.reader(open(OUTFILE), delimiter='\t'):
        if validation:
            if line[1] in trainIDs:
                continue
        if line[3] == 'YES' or line[3] == 'NO':
            i += 1
            if line[3] == 'YES':
                y += 1

    print 'We answered YES in %.2f%% of answered (%d)' % (y/i*100,i)

def bad_only():
    BADFILE = 'tests/bad_outfile.tsv'
    with open(BADFILE, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        for line in csv.reader(open(OUTFILE), delimiter='\t'):
            if line[2] != line[3]:
                writer.writerow(line)


import sys
CSVFOLDER = "tests/batches"
OUTFILE = "tests/outfile.tsv"
INFOFILE = "tests/infofile.tsv"
validation = True
if __name__ == "__main__":
    for i in range(0,len(sys.argv)):
        if sys.argv[i] == '-train':
            CSVFOLDER += '/batch_train'
            OUTFILE = "tests/outfile_train.tsv"
        if sys.argv[i] == '-test':
            CSVFOLDER += '/batch_test'
            OUTFILE = "tests/outfile_test.tsv"
        if sys.argv[i] == '-valoff':
            validation = False
    reparse()
    get_stats()
    print '----------'
    turkstats()
    print '----------'
    more_stats()
    bad_only()
