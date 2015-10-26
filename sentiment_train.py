from __future__ import division
# -*- coding: utf-8 -*-
import csv
import numpy as np
from sklearn import linear_model
from sklearn.externals import joblib

class QuestionFeatures(object):
    def __init__(self):
        self.q = []
        self.s = []
        self.verb_sim = []
        self.ans = []
        self.ID = ''

numfeatures = 3
trainIDs = []
def load():
    qf = []
    i = 0
    sp = 0
    ansp = 0
    vsp = 0
    for line in csv.reader(open(OUTFILE), delimiter='\t'):
        if i == 0:
            i += 1
            for j in range(0,len(line)):
                if line[j] == 'Sentiment':
                    sp = j
                if line[j] == 'Verb Similarity':
                    vsp = j
                if line[j] == 'TurkAnswer':
                    ansp = j
            continue
        i += 1
        if i % 2 == 1:
            trainIDs.append(line[1])
        if line[sp] == '':
            continue

        sentiment = line[sp].split(":")
        verb_sim = line[vsp].split(":")
        f = QuestionFeatures()
        f.ID = line[1]
        for j in range(0,len(sentiment)):
            sent = sentiment[j].split()
            f.q.append(int(sent[0]))
            f.s.append(int(sent[1]))
            f.verb_sim.append(float(verb_sim[j]))
#            f.h.append(int(sentiment[2]))
            if line[ansp] == 'YES':
                f.ans.append(1)
            else:
                f.ans.append(0)
        qf.append(f)

    return qf

def split_train(qf):
    trainqf = []
    testqf = []
    for i in range(0,len(qf)):
        if validation:
            if qf[i].ID in trainIDs:
                trainqf.append(qf[i])
            else:
                testqf.append(qf[i])
        else:
            trainqf.append(qf[i])
            testqf.append(qf[i])
    tesx,tesy = fill(testqf)
    trax,tray = fill(trainqf)
    return (tesx,tesy,trax,tray)

def fill(qf):
    q = []
    s = []
    sim = []
    ans = []
    for f in qf:
        q += f.q
        s += f.s
        sim += f.verb_sim
        ans += f.ans
    q = np.array(q)
    s = np.array(s)
    sim = np.array(sim)
    y = np.array(ans)
    x = np.vstack((q, s, sim)).transpose()
    return x,y

def even_out(x,y):
    indexes = []
    while sum(y) < len(y)/2:
        added = 0
        for i in range(0,len(y)):
            if y[i] == 1 and i not in indexes:
                y = np.hstack((y,y[i]))
                x = np.vstack((x,x[i]))
                indexes.append(i)
                added += 1
                break
        if added == 0:
            indexes = []

    while sum(y) > len(y)/2:
        added = 0
        for i in range(0,len(y)):
            if y[i] == 0 and i not in indexes:
                y = np.hstack((y,y[i]))
                x = np.vstack((x,x[i]))
                indexes.append(i)
                added += 1
                break
        if added == 0:
            indexes = []
    return x,y

def train(qf):
#    clf = joblib.load('sources/models/sentiment.pkl')
    clf = linear_model.LogisticRegression(C=1, penalty='l2', tol=1e-4)

    xtest,ytest,xtrain,ytrain = split_train(qf)

    xtrain,ytrain = even_out(xtrain,ytrain)
    clf.fit(xtrain, ytrain)

    joblib.dump(clf, 'sources/models/sentiment.pkl')
    counttest=clf.predict_proba(xtest)[:,1]

    correct = 0
    for i in range(0,len(ytest)):
        if counttest[i] < 0.5:
            an = 0
        else:
            an = 1
        if an == ytest[i]:
            correct += 1

#    for i in range(0,len(y)):
#        if counttest[i] < 0.5:
#            an = 0
#        else:
#            an = 1
#        if an == y[i]:
#            correct += 1
    print 'train: yes = %.2f%%' % (sum(ytrain)/len(ytrain)*100)
    print 'test: yes = %.2f%%' % (sum(ytest)/len(ytest)*100)
    print 'Correct %.2f%% from test' % (correct/len(ytest)*100)
#    print 'New sentiment correct %.2f%%' % (correct/len(y)*100)
    w = clf.coef_
    w = np.append(w, clf.intercept_);
    print w

def saveIDs():
    ids = np.array(trainIDs)
    np.save('tests/trainIDs/trainIDs.npy',ids)

validation = False
OUTFILE = 'tests/outfile.tsv'
import sys
validation = True
if __name__ == "__main__":
    for i in range(0,len(sys.argv)):
        if sys.argv[i] == '-train':
            OUTFILE = "tests/outfile_train.tsv"
        if sys.argv[i] == '-test':
            OUTFILE = "tests/outfile_test.tsv"
        if sys.argv[i] == '-val':
            validation = True
    qf = load()
    train(qf)
    saveIDs()
