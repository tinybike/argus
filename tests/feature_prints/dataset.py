"""Extract question|sentence|gs class"""

import csv
import numpy as np

trainIDs = np.load('../trainIDs/trainIDs.npy')

table_test = []
table_train = []
for line in csv.reader(open('all_features.tsv'), delimiter='\t'):
    if line[2] not in '10':
        continue
    if line[0] in trainIDs:
        table_train.append([line[0], line[2], line[1].replace('\n', ' ')])
    else:
        table_test.append([line[0], line[2], line[1].replace('\n', ' ')])


writer = csv.writer(open('argus_train.csv', 'wb'), delimiter=',')
writer.writerow(['qtext', 'label', 'atext'])
for triplet in table_train:
    writer.writerow(triplet)

writer = csv.writer(open('argus_test.csv', 'wb'), delimiter=',')
writer.writerow(['qtext', 'label', 'atext'])
for triplet in table_test:
    writer.writerow(triplet)
