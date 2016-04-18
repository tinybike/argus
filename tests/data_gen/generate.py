"""
Autogenerates sports and politics questions from an event database.
"""

import csv
import json
import random
random.seed(1234567)


def split(i):
    # val_split and test_split are 1/5 both
    splits = [
            'train',
            'val',
            'train',
            'train',
            'train',
            'test',
            'train',
            'train',
            'train',
            'train',
            ]
    return splits[i % len(splits)]


def replace(sentence, old_l, new_l):
    for (old, new) in zip(old_l, new_l):
        sentence = sentence.replace(old, new)
    return sentence


def load_sport(fname):
    eventsets = json.load(open(fname))
    for eventset in eventsets:
        for event in eventset['events']:
            yield (eventset, event)


def sport_questions(eventset, event):
    bonus_sport = ["will <team> win the <event> <year> finals?",
                   "will <team> win the <event> <year> semi-finals?",
                   "will <team> win the <event> <year> quarter-finals?",
                   "will <team> qualify for the <event> <year>?",
                   "will <team> win the <event> finals?",
                   "will <team> win the <event> semi-finals?",
                   "will <team> win the <event> quarter-finals?",
                   "will <team> qualify for the <event>?",
                   "did <team> win the <event> <year> finals?",
                   "did <team> win the <event> <year> semi-finals?",
                   "did <team> win the <event> <year> quarter-finals?",
                   "did <team> qualify for the <event> <year>?",
                   "did <team> win the <event> finals?",
                   "did <team> win the <event> semi-finals?",
                   "did <team> win the <event> quarter-finals?",
                   "did <team> qualify for the <event>?"]

    for team in eventset['teams']:
        old = ['<event>', '<year>', '<team>']
        new = [eventset['event_name'], event['year'], team]
        if eventset['event_type'] == 'team':
            new[2] = "the "+new[2]

        for sentence in open('sport_sentences.txt'):
            question = sentence[:-1]

            q = replace(question, old, new)
            if team in event['winner']:
                ans = 'YES'
            else:
                ans = 'NO'
                if random.random() > 0.8:
                    continue
            yield (q, ans)

        if team in event['winner']:
            ans = 'YES'
            for q1 in bonus_sport:
                q1 = replace(q1, old, new)
                yield (q1, ans)


def load_politics(fname):
    ev = []
    names = []
    for line in open(fname):
        new = line.split('\n')[0].split('\t')
        names.append(new[1])
        ev.append(new)
    return (ev, names)


def pol_questions(ev, names):
    for sentence in open('politics_sentences.txt'):
        question = sentence[:-1]

        old = ['<state>', '<name>', '<position>', '<election>']
        q = replace(question, old, ev)
        ans = 'YES'
        yield (q, ans)

        while True:
            name = random.choice(names)
            if name != ev[1]:
                break
        ev_n = list(ev)
        ev_n[1] = name
        old = ['<state>', '<name>', '<position>', '<election>']
        q = replace(question, old, ev_n)
        ans = 'NO'
        yield (q, ans)


if __name__ == "__main__":
    qtsv = {}
    for k in ['train', 'val', 'test']:
        qtsv[k] = csv.writer(open('../q%s.tsv' % (k,), 'ab'), delimiter='\t')

    sport_events = load_sport('sport.json')
    for i, ev in enumerate(sport_events):
        for q, ans in sport_questions(*ev):
            qtsv[split(i)].writerow(['gen', i, 'sport', q, ans, '*'])
    i0 = i

    pol_events, pol_names = load_politics('politics')
    for i, ev in enumerate(pol_events):
        for q, ans in pol_questions(ev, pol_names):
            qtsv[split(i)].writerow(['gen', i0 + i, 'politics', q, ans, '*'])
