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
    qp = 0
    sp = 0
    hp = 0
    ansp = 0
    for line in csv.reader(open('tests/outfile.tsv'), delimiter='\t'):
        if i == 0:
            i += 1
            for j in range(0,len(line)):
                if line[j] == 'QSentiment':
                    qp = j
                if line[j] == 'SSentiment':
                    sp = j
                if line[j] == 'HSentiment':
                    hp = j
                if line[j] == 'TurkAnswer':
                    ansp = j
        else:
            if line[qp] == '0' and line[sp] == '0' and line[hp] == '0':
                continue
            q.append(int(line[qp]))
            s.append(int(line[sp]))
            h.append(int(line[hp]))
            if line[ansp] == 'YES':
                ans.append(1)
            else:
                ans.append(0)

    q = np.array(q)
    s = np.array(s)
    h = np.array(h)
    y = np.array(ans)
    x = np.vstack((q, s, h)).transpose()
    return (x,y)

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
    print 'New sentiment correct %.2f%%' % (correct/len(ytest)*100)
#    print 'New sentiment correct %.2f%%' % (correct/len(y)*100)
    w = clf.coef_
    w = np.append(w, clf.intercept_);
    print w

if __name__ == "__main__":
    x,y = load()
    train(x,y)