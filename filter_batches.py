# -*- coding: utf-8 -*-
import csv
import os
OUTFILE = 'tests/batches/filtered.csv'
CSVFOLDER = 'tests/batches/origin'
def filter_rejected():
    qnum = 0
    with open(OUTFILE, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for csvfile in os.listdir(CSVFOLDER):
            i = 0
            for line in csv.reader(open(CSVFOLDER+'/'+csvfile), delimiter=',',skipinitialspace=True):
                if i == 0 and qnum != 0:
                    i += 1
                    qnum += 1
                    continue
                i += 1
                qnum += 1
                if line[16] == 'Rejected':
                    continue
                writer.writerow(line)

filter_rejected()