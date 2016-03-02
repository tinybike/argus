"""
generates sports and politics dataset, dumps to mturk and dataset-sts formats
"""
import json
import csv
import random
import numpy as np
random.seed(1234567)


def replace(sentence, old_l, new_l):
    for (old, new) in zip(old_l, new_l):
        sentence = sentence.replace(old, new)
    return sentence

sport = json.load(open('sport.json'))
i = 0
y = 0.


questions = []

bonus_sport = ["Will <team> win the <event> <year> finals?",
               "Will <team> win the <event> <year> semi-finals?",
               "Will <team> win the <event> <year> quarter-finals?",
               "Will <team> qualify for the <event> <year>?",
               "Will <team> win the <event> finals?",
               "Will <team> win the <event> semi-finals?",
               "Will <team> win the <event> quarter-finals?",
               "Will <team> qualify for the <event>?",
               "Did <team> win the <event> <year> finals?",
               "Did <team> win the <event> <year> semi-finals?",
               "Did <team> win the <event> <year> quarter-finals?",
               "Did <team> qualify for the <event> <year>?",
               "Did <team> win the <event> finals?",
               "Did <team> win the <event> semi-finals?",
               "Did <team> win the <event> quarter-finals?",
               "Did <team> qualify for the <event>?"]


for sentence in open('sport_sentences.txt'):
    question = sentence[:-1]
    for sport_event in sport:
        for event in sport_event['events']:
            winner = event['winner']
            for team in sport_event['teams']:
                old = ['<event>', '<year>', '<team>']
                new = [sport_event['event_name'], event['year'], team]
                if sport_event['event_type'] == 'team':
                    new[2] = "the "+new[2]
                q = replace(question, old, new)
                if team in winner:
                    ans = 'YES'
                    y += 1
                    for q1 in bonus_sport:
                        q1 = replace(q1, old, new)
                        y += 1
                        i += 1
                        questions.append((q1, ans))
                        print q1, ans
                else:
                    ans = 'NO'
                    if random.random() > 0.8:
                        continue
                questions.append((q, ans))
                print q, ans
                i += 1

for sentence in open('politics_sentences.txt'):
    question = sentence[:-1]
    names = []
    for line in open('politics'):
        new = line.split('\n')[0].split('\t')
        names.append(new[1])
        old = ['<state>', '<name>', '<position>', '<election>']
        q = replace(question, old, new)
        ans = 'YES'
        y += 1
        questions.append((q, ans))
        print q, ans
        i += 1

    for line in open('politics'):
        new = line.split('\n')[0].split('\t')
        while True:
            name = random.choice(names)
            if name != new[1]:
                break
        new[1] = name
        old = ['<state>', '<name>', '<position>', '<election>']
        q = replace(question, old, new)
        ans = 'NO'
        questions.append((q, ans))
        print q, ans
        i += 1

print i
print y/i
used = []
with open('../batches/generated.csv', 'wb') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['*']*32)
    for q, ans in questions:
        if q in used:
            continue
        else:
            used.append(q)
        line = ['*']*28
        line += [ans, '*', q, '*']
        writer.writerow(line)


def sts_out():
    trainIDs = list(np.load('../trainIDs/trainIDs.npy'))

    writer_train = csv.writer(open('argus_train.csv', 'wb'), delimiter=',')
    writer_test = csv.writer(open('argus_test.csv', 'wb'), delimiter=',')
    writer_train.writerow(['qtext', 'label', 'atext'])
    writer_test.writerow(['qtext', 'label', 'atext'])
    for line in csv.reader(open('../feature_prints/all_features.tsv'), delimiter='\t'):
        if line[2] == 'YES':
            ans = 1
        elif line[2] == 'NO':
            ans = 0
        else:
            continue
        if line[0] in trainIDs:
            writer_train.writerow([line[0], ans, line[1].rstrip()])
        else:
            writer_test.writerow([line[0], ans, line[1].rstrip()])


sts_out()