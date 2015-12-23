# -*- coding: utf-8 -*-
import numpy as np
import csv
from sklearn import linear_model
from sklearn import svm

def train():
    z = np.load('tests/batches/relevance/npy_rel.npy')
    print z.shape
    x = z[:,:-1]
    y = z[:,-1]
    print float(sum(y))/len(y)
    clf = linear_model.LogisticRegression(C=1, penalty='l2', tol=1e-5)
    # clf = svm.SVC(kernel='linear', C=1, probability=True)
    clf.fit(x, y)
    test = clf.predict_proba(x)[:,1]
    corr = 0
    fp = 0
    fn = 0
    for i in range(len(test)):
        if (y[i] == 1 and test[i] > 0.5) or (y[i] == 0 and test[i] < 0.5):
            corr += 1.
        elif y[i] == 1:
            fn += 1.
        elif y[i] == 0:
            fp += 1.
    print 'accuracy %.3f, false positives %.3f, false negatives %.3f' % (
        corr/len(y), fp/(fn+fp), fn/(fn+fp)
        )
#    test = abs(y-test) < 0.5
#    print float(sum([int(k) for k in test]))/len(test)

    w = clf.coef_
    w = np.append(w,clf.intercept_);
    np.save('tests/batches/relevance/learned_relevance',w)
    print w

def relevance_load():
    relevance = []
    sentences = []
    questions = []
    rels = []
    i = 0
    for line in csv.reader(open('tests/batches/relevance/relevance.csv'), delimiter=',',skipinitialspace=True):
        if i == 0:
            i += 1
            for word in line:
                if 'question' in word:
                    questions.append(line.index(word))
                if 'sentence' in word:
                    sentences.append(line.index(word))
                if 'relevance' in word:
                    rels.append(line.index(word))
            continue
        if sum([int(line[ix]) for ix in rels]) == 10:
            continue
        for ix in range(len(rels)):
            rel = int(line[rels[ix]])
            if rel != 1:
                relevance.append([line[questions[ix]], line[sentences[ix]], rel])
            else:
                relevance.append([line[questions[ix]], line[sentences[ix]], 0])
    return relevance


def filter_sources(answer):
    r = relevance_load()
    newsources = []
    for triplet in r:
        if answer.q.text == triplet[0]:
            for s in answer.sources:
                if s.sentence == triplet[1]:
                    if triplet[2] == 2:
                        newsources.append(s)
    answer.sources = newsources

if __name__ == '__main__':
    train()
