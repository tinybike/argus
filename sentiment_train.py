from __future__ import division
# -*- coding: utf-8 -*-
import csv
import numpy as np
from sklearn import linear_model
from sklearn.externals import joblib
q = []
s = []
h = []
ans = []
i = 0
qp = 0
sp = 0
hp = 0
ansp = 0
for line in csv.reader(open('tests/outfile (copy).tsv'), delimiter='\t'):
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
correct = 0
for i in range(0,len(q)):
    a = s[i]+h[i]
    if q[i] == 0:
        if a < 0:
            an = 0
        else:
            an = 1
    elif q[i] > 0:
        if a < 0:
            an = 0
        else:
            an = 1
    elif q[i] < 0:
        if a < 0:
            an = 1
        else:
            an = 1
    print an,y[i]
    if an == y[i]:
        correct += 1

print 'Basic sentiment correct %.2f%%' % (correct/len(q)*100)


x = np.vstack((q, s, h)).transpose()



clf = linear_model.LogisticRegression(C=1, penalty='l2', tol=1e-4)
clf.fit(x, y)

joblib.dump(clf, 'sources/models/sentiment.pkl')
counttest=clf.predict_proba(x)[:,1]

correct = 0
for i in range(0,len(q)):
    if counttest[i] < 0.5:
        an = 0
    else:
        an = 1
    if an == y[i]:
        correct += 1

print 'New sentiment correct %.2f%%' % (correct/len(q)*100)
w = clf.coef_
w = np.append(w, clf.intercept_);
print w
