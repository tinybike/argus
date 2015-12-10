from __future__ import division
import csv
import os
from argus.main_frame import get_answer
import numpy as np
from argus.features import feature_list_official as flo


CSVFOLDER = "tests/batches"
trainIDs = np.load('tests/trainIDs/trainIDs.npy')

first = False
def reparse():
    qnum = 0
    info_files = [open('tests/feature_prints/'+i_f+'.tsv', 'wb') for i_f in flo]
    writers = [csv.writer(info_file, delimiter='\t') for info_file in info_files]
    info_all = open('tests/feature_prints/all_features.tsv', 'wb')
    writer_all = csv.writer(info_all, delimiter='\t')
    info_turk = open('tests/feature_prints/turk_sentences.tsv', 'wb')
    writer_turk = csv.writer(info_turk, delimiter=',')
    first = False
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
                    info += flo
                    if qnum == 0:
                        writer.writerow(info)
                        writer_turk.writerow(['question', 'sentence'])
                        first = True
                    continue
                if line[16] == 'Rejected':
                    continue
                qnum += 1
                ouranswer = get_answer(line[30])

                url = ''
                headline = ''
                sentence = ''
                source = ''
                feat = ''
                info = []
                if len(ouranswer.sources) != 0:
                    feature_print_all(writer_all, ouranswer, first)
                    feature_print(writers, ouranswer)
                    turk_print(writer_turk, ouranswer)
                    url = ouranswer.sources[0].url
                    headline = ouranswer.sources[0].headline
                    sentence = ouranswer.sources[0].sentence
                    source = ouranswer.sources[0].source
                    for j in range(len(flo)):
                        for s in ouranswer.sources:
                            feat += str(s.features[j].get_value())+":"
                        feat = feat[:-1]
                        info.append(feat)
                        feat = ''
                info = [line[0], line[30], line[28],
                        ouranswer.text, ouranswer.q.query, sentence,
                        headline, line[31], line[29], url, source,
                        ouranswer.info] + info
                info = [field.encode('utf-8') for field in info]
                writer.writerow(info)
                if qnum % 10 == 0:
                    print 'answering question', qnum
    for i_f in info_files:
        i_f.close()


def feature_print(writers, answer):
    for source in answer.sources:
        for i in range(len(writers)):
            f = source.features[i]
            info = [answer.q.text, f.get_info(), str(f.get_value()), source.sentence]
            info = [field.encode('utf-8') for field in info]
            writers[i].writerow(info)

def turk_print(writer, answer):
    for source in answer.sources:
        info = [answer.q.text, source.sentence]
        info = [field.encode('utf-8') for field in info]
        writer.writerow(info)

def feature_print_all(writer, answer, first = False):
    if first:
        writer.writerow(['Question'] + [f.get_name() for f in answer.sources[0].features]+['Sentence'])
        first = False
    for source in answer.sources:
        info = [answer.q.text] + [str(f.get_value()) for f in source.features] + [source.sentence]
        info = [field.encode('utf-8') for field in info]
        writer.writerow(info)


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
