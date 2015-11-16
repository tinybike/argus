# -*- coding: utf-8 -*-
import csv
import os

def filter_rejected():
    OUTFILE = 'mturkdate.csv'
    CSVFOLDER = 'batches'
    qnum = 0
    with open(OUTFILE, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for csvfile in os.listdir(CSVFOLDER):
            if not csvfile.endswith('.csv'):
                continue
            i = 0
            for line in csv.reader(open(CSVFOLDER+'/'+csvfile), delimiter=',',skipinitialspace=True):
                if i == 0 and qnum != 0:
                    i += 1
                    qnum += 1
                    continue
                if line[16] == 'Rejected':
                    continue
                info = [line[30],line[29]]
                if i == 0 and qnum == 0:
                    info = ['question','url']
                writer.writerow(info)
                i += 1
                qnum += 1

from datetime import date
class Question_Info(object):
    def __init__(self,info):
        self.question = info[0]
        self.info = [info]

    def add(self, info):
        self.info.append(info)

    def decide(self):
        if len(self.info) < 3:
            return False,[]
        ok = []
        wrong = []
        dates = []
        for info in self.info:
            ok.append(int(info[2]))
            wrong.append(info[3].lower())
            try:
                dates.append(date(int(info[6]),int(info[5]),int(info[4])))
            except ValueError:
                dates.append(date(randint(1,2015),1,1))
        if sum(ok) < 2:
            return False, []
        if wrong.count('no date') > 1:
            return False, []

        if dates[0] == dates[1]:
            return True, dates[0]
        if dates[0] == dates[2]:
            return True, dates[0]
        if dates[2] == dates[1]:
            return True, dates[2]
        return False, []


def date_merge():
    OUTFILE = 'filtereddate.tsv'
    CSVFILE = 'Batch_2160587_batch_results.csv'

    i = 0
    inout = ['Input.question', 'Input.url',  'Answer.ok', 'Answer.web_url', 'Answer.day', 'Answer.month', 'Answer.year']
    QIS = []
    for line in csv.reader(open(CSVFILE), delimiter=',',skipinitialspace=True):
        if i == 0:
            i += 1
            inoutpos = [line.index(x) for x in inout]
            continue
        info = [line[x] for x in inoutpos]
        if i == 1:
            i += 1
            QIS.append(Question_Info(info))
            continue

        if info[0] == QIS[-1].question:
            QIS[-1].add(info)
        else:
            QIS.append(Question_Info(info))

    with open(OUTFILE, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        ok = 0
        ko = 0
        for qi in QIS:
            found, d = qi.decide()
            if found:
                ok += 1
                writer.writerow([qi.question, d])
    #            print 'date found:', qi.question, date
            else:
                ko += 1
    print 'found %d dates out of %d' % (ok, ok+ko)

if __name__ == "__main__":
    date_merge()