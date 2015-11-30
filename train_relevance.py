# -*- coding: utf-8 -*-
import numpy as np
from argus.relevance import Relevance, Q
import csv
outfile = 'tests/outfile.tsv'
trainIDs = np.load('tests/trainIDs/trainIDs.npy')

def fill():
    clas = []
    rel = []
    ctext = []
    rtext = []
    ta = 0
    i = 0
    qstest = []
    qstrain = []
    for line in csv.reader(open(outfile), delimiter='\t',skipinitialspace=True):
        if i == 0:
            i += 1
            for field in line:
                if field == 'TurkAnswer':
                    ta = line.index(field)
                if '#' in field:
                    clas.append(line.index(field))
                    ctext.append(field)
                if '@' in field:
                    rel.append(line.index(field))
                    rtext.append(field)
            continue
        if len(line) <= clas[0]:
            continue
        r = []
        f = []
        for index in clas:
            fi = np.array([float(x) for x in line[index].split(':')])
            if len(f) == 0:
                f = fi
            else:
                f = np.vstack((f, fi))
        for index in rel:
            ri = np.array([float(x) for x in line[index].split(':')])
            if len(r) == 0:
                r = ri
            else:
                r = np.vstack((r, ri))

        if line[ta] == 'YES':
            y = 1
        else:
            y = 0
        if line[1] in trainIDs:
            qstrain.append(Q(f,r,y))
        else:
            qstest.append(Q(f,r,y))
    return qstrain, qstest, ctext, rtext

def stats(R, qs):
    corr = 0
    i = 0
    for q in qs:
        i += 1
        yt = R.forward_propagation(q.f, q.r)
        if yt > 0.5:
            yt = 1
        else:
            yt = 0
        if q.y == yt:
            corr += 1
    print '%.2f%% correct (%d/%d)' % (float(corr)/i*100, corr, i)

def multip_features(qs, ctext=None, rtext=None):
    k = 0
    for q in qs:
        flen = len(q.f)
        for i in range(flen):
            for j in range(flen):
                if i >= j:
                    continue
                newf = q.f[i,:]*q.f[j,:]
                q.f = np.vstack((q.f, newf))
                if k == 0 and ctext is not None:
                    ctext.append(ctext[i]+'_X_'+ctext[j])
        k = 1
    k = 0
    for q in qs:
        rlen = len(q.r)
        for i in range(rlen):
            for j in range(rlen):
                if i >= j:
                    continue
                newr = q.r[i,:]*q.r[j,:]
                q.r = np.vstack((q.r, newr))
                if k == 0 and rtext is not None:
                    rtext.append(rtext[i]+'_X_'+rtext[j])
        k = 1


def list_weights(R, ctext, rtext):
    for i in range(len(ctext)):
        dots = max(3,50-len(ctext[i]))
        print '(class) %s%s%.2f' % (ctext[i], '.'*dots, R.W[i])
    print '(class) bias........%.2f' % (R.W[-1])
    for i in range(len(rtext)):
        dots = max(3,50-len(rtext[i]))
        print '(rel) %s%s%.2f' % (rtext[i], '.'*dots, R.Q[i])
    print '(rel) bias........%.2f' % (R.Q[-1])

def train():
    qstrain, qstest, ctext, rtext = fill()
#    multip_features(qstrain, ctext, rtext)
#    multip_features(qstest)
    w_dim = qstest[0].f.shape[0]
    q_dim = qstest[0].r.shape[0]
    R = Relevance(w_dim, q_dim)
    for i in range(50):
        R.train(qstrain, learning_rate=0.01, nepoch=10, evaluate_loss_after=10,
            batch_size=200, reg=1e-3)
        print '---------------test'
        stats(R, qstest)
        print '---------------train'
        stats(R, qstrain)
        print i

    R.save('sources/models')
    print '---------------test'
    stats(R, qstest)
    print '---------------train'
    stats(R, qstrain)
    print '\n========================\n'
    list_weights(R, ctext, rtext)


if __name__ == '__main__':
    np.random.seed(17151711)
    train()