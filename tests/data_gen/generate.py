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
    for team in eventset['teams']:
        old = ['<event>', '<year>', '<team>']
        new = [eventset['event_name'], event['year'], team]
        if eventset['event_type'] == 'team':
            new[2] = "the "+new[2]

        for sentence in open('sport_sentences.txt'):
            question = sentence[:-1]

            if question.startswith('@'):
                # winner-only question
                if team not in event['winner']:
                    continue
                question = question[1:]

            if question.startswith('!'):
                flip_ans = True
                question = question[1:]
            else:
                flip_ans = False

            q = replace(question, old, new)
            if team in event['winner']:
                ans = True
            else:
                ans = False
                if random.random() > 0.8:
                    continue

            if flip_ans:
                ans = not ans
            yield (q, 'YES' if ans else 'NO')


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

        if question.startswith('@'):
            winner_only = True
            question = question[1:]
        else:
            winner_only = False
        if question.startswith('!'):
            flip_ans = True
            question = question[1:]
        else:
            flip_ans = False

        old = ['<state>', '<name>', '<position>', '<election>']
        q = replace(question, old, ev)
        ans = 'YES' if not flip_ans else 'NO'
        yield (q, ans)

        if winner_only:
            continue

        while True:
            name = random.choice(names)
            if name != ev[1]:
                break
        ev_n = list(ev)
        ev_n[1] = name
        old = ['<state>', '<name>', '<position>', '<election>']
        q = replace(question, old, ev_n)
        ans = 'NO' if not flip_ans else 'YES'
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
