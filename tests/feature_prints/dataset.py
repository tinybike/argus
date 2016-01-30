"""Extract question|sentence|gs class"""

import csv

table = []
i = 0
for line in csv.reader(open('all_features.tsv'), delimiter='\t'):
    i+=1
    if len(line)<3:
        print line, i
    if line[2] not in '10':
        continue
    table.append([line[0], line[2], line[1]])

writer = csv.writer(open('argus_gen.csv', 'wb'), delimiter=',')
writer.writerow(['qtext', 'label', 'atext'])
for triplet in table:
    writer.writerow(triplet)
