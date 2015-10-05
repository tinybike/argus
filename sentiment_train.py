from __future__ import division
# -*- coding: utf-8 -*-
import csv
import numpy as np
from sklearn import linear_model
from sklearn.externals import joblib

def load():
    q = []
    s = []
    h = []
    ans = []
    i = 0
    sp = 0
    ansp = 0
    for line in csv.reader(open('tests/outfile (copy).tsv'), delimiter='\t'):
        if i == 0:
            i += 1
            for j in range(0,len(line)):
                if line[j] == 'Sentiment':
                    sp = j
                if line[j] == 'TurkAnswer':
                    ansp = j
        else:
            if line[sp] == '':
                continue
            sources = line[sp].split(":")
            for triplet in sources:
                sentiment = triplet.split()
                q.append(int(sentiment[0]))
                s.append(int(sentiment[1]))
                h.append(int(sentiment[2]))
                if line[ansp] == 'YES':
                    ans.append(1)
                else:
                    ans.append(0)


    q,s,h,ans = filter_yes(q,s,h,ans)

    q = np.array(q)
    s = np.array(s)
    h = np.array(h)
    y = np.array(ans)
    print len(y)
    x = np.vstack((q, s, h)).transpose()
    return (x,y)


def filter_yes(q,s,h,y):
    q2 = []
    s2 = []
    h2 = []
    y2 = []
    for i in range(0,len(y)):
        if y[i] == 1:
            if rand() > 0.3:
                continue
        q2.append(q[i])
        s2.append(s[i])
        h2.append(h[i])
        y2.append(y[i])
    return q2,s2,h2,y2

def train(x,y):
#    clf = joblib.load('sources/models/sentiment.pkl')
    clf = linear_model.LogisticRegression(C=1, penalty='l2', tol=1e-4)
    testsize = int(len(y)/2)
    trainsize = len(y)-testsize
    xtrain = np.zeros((trainsize,3))
    ytrain = np.zeros(trainsize)
    xtest = np.zeros((testsize,3))
    ytest = np.zeros(testsize)
    for i in range(0, len(y)):
        if i % 2 == 0:
            ytrain[int(i/2)] = y[i]
            xtrain[int(i/2),:] = x[i,:]
        else:
            ytest[int(i/2)] = y[i]
            xtest[int(i/2),:] = x[i,:]

    clf.fit(xtrain, ytrain)
#    clf.fit(x, y)

    joblib.dump(clf, 'sources/models/sentiment.pkl')
    counttest=clf.predict_proba(xtest)[:,1]
#    counttest=clf.predict_proba(x)[:,1]

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
    print 'train: yes = %.2f%%' % (sum(ytrain)/len(ytrain))
    print 'test: yes = %.2f%%' % (sum(ytest)/len(ytest))
    print 'Correct %.2f%% from test' % (correct/len(ytest)*100)
#    print 'New sentiment correct %.2f%%' % (correct/len(y)*100)
    w = clf.coef_
    w = np.append(w, clf.intercept_);
    print w

if __name__ == "__main__":
    x,y = load()
    train(x,y)