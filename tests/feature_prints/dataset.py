#!/usr/bin/python
"""Extract question|sentence|gs class from per-pair feature dumps
for dataset-sts datasets.

Usage: tests/feature_prints/dataset.py tests/feature_prints/train/all_features.tsv >argus_train.csv """

import csv
import numpy as np
import sys

allftsv = sys.argv[1]

writer = csv.writer(sys.stdout, delimiter=',')
writer.writerow(['qtext', 'label', 'atext'])

for line in csv.reader(open(allftsv), delimiter='\t'):
    gs = line[2]
    if gs == 'YES':
        gs = 1
    elif gs == 'NO':
        gs = 0
    else:
        continue
    writer.writerow([line[0], gs, line[1].replace('\n', ' ')])
