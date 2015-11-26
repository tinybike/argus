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
            print ctext,rtext
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

def train():
    qstrain, qstest, ctext, rtext = fill()
    w_dim = qstest[0].f.shape[0]
    q_dim = qstest[0].r.shape[0]
    R = Relevance(w_dim, q_dim)
    R.train(qstrain, learning_rate=0.01, nepoch=500, evaluate_loss_after=10,
            batch_size=200, reg=1e-3)
    R.save('sources/models')
    print '---------------test'
    stats(R, qstest)
    print '---------------train'
    stats(R, qstrain)
    print '\n========================\n'
    print ' '.join(ctext)
    print R.W
    print ' '.join(rtext)
    print R.Q

if __name__ == '__main__':
    np.random.seed(17151711)
    train()