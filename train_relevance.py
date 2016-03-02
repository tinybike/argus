# -*- coding: utf-8 -*-
"""
Training Relevance model happens here, you can change various parameters in train().
"""
import numpy as np
from argus.relevance import Relevance, Q
import csv
import sys
from multiprocessing import Pool

outfile = 'tests/outfile.tsv'
trainIDs = []


def saveIDs():
    ids = np.array(trainIDs)
    np.save('tests/trainIDs/trainIDs.npy', ids)


def train():
    qstrain, qstest, ctext, rtext = load_features()
    # inverse_features(qstrain, ctext, rtext)
    # inverse_features(qstest)
    # multip_features(qstrain, ctext, rtext)
    # multip_features(qstest)
    zero_features(qstrain, ctext, rtext)
    zero_features(qstest)

    R = cross_validate_all(qstrain+qstest)

    # R = Relevance(qstest[0].f.shape[0], qstest[0].r.shape[0])
    # R.Q = np.zeros_like(R.Q)
    # R.W = np.zeros_like(R.W)
    # R.W[-2] = 1.
    # R.Q[-1] = 1.
    # R.load('sources/models')
    # R.train(qstrain+qstest, learning_rate=0.02, nepoch=20, evaluate_loss_after=10,
    #         batch_size=10, reg=1e-3)

    print '\n========================\n'
    list_weights(R, ctext, rtext)
    print 'W_shape =', R.W.shape
    print 'Q_shape =', R.Q.shape
    print '---------------test'
    stats(R, qstest)
    print '---------------train'
    stats(R, qstrain)
    if query_yes_no('Save model and rewrite output.tsv?'):
        R.save('sources/models')
        rewrite_output()


def load_features():
    clas = []
    rel = []
    ctext = []
    rtext = []
    ta = 0
    i = 0
    qstest = []
    qstrain = []
    for line in csv.reader(open(outfile), delimiter='\t', skipinitialspace=True):
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
        i += 1
        if split(i):
            trainIDs.append(line[1])
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
        if split(i):
            qstrain.append(Q(line[1], f, r, y))
        else:
            qstest.append(Q(line[1], f, r, y))

    saveIDs()
    return qstrain, qstest, ctext, rtext


def split(i):
    return i % 4 < 2

def multip_features(qs, ctext=None, rtext=None):
    """ extend FV with feature powerset (f1*f2 for all f1, f2) metafeatures """
    k = 0
    for q in qs:
        flen = len(q.f)
        for i in range(flen):
            for j in range(flen):
                if i >= j:
                    continue
                newf = q.f[i, :] * q.f[j, :]
                q.f = np.vstack((q.f, newf))
                if k == 0 and ctext is not None:
                    ctext.append(ctext[i] + '_X_' + ctext[j])
        k = 1
    k = 0
    for q in qs:
        rlen = len(q.r)
        for i in range(rlen):
            for j in range(rlen):
                if i >= j:
                    continue
                newr = q.r[i, :] * q.r[j, :]
                q.r = np.vstack((q.r, newr))
                if k == 0 and rtext is not None:
                    rtext.append(rtext[i] + '_X_' + rtext[j])
        k = 1


def zero_features(qs, ctext=None, rtext=None):
    """ extend FV with <feature>==0 metafeatures """
    k = 0
    for q in qs:
        flen = len(q.f)
        for i in range(flen):
            newf = q.f[i, :] == 0.
            q.f = np.vstack((q.f, newf.astype(float)))
            if k == 0 and ctext is not None:
                ctext.append(ctext[i] + '==0')
        k = 1


def inverse_features(qs, ctext=None, rtext=None):
    """ extend FV with 1-<feature> metafeatures """
    k = 0
    for q in qs:
        flen = len(q.f)
        for i in range(flen):
            newf = q.f[i, :] - 1
            q.f = np.vstack((q.f, newf))
            if k == 0 and ctext is not None:
                ctext.append('1-' + ctext[i])
        k = 1


def list_weights(R, ctext, rtext):
    for i in range(len(ctext)):
        dots = max(3, 50 - len(ctext[i]))
        print '(class) %s%s%.2f' % (ctext[i], '.' * dots, R.W[i])
    print '(class) bias........%.2f' % (R.W[-1])
    for i in range(len(rtext)):
        dots = max(3, 50 - len(rtext[i]))
        print '(rel) %s%s%.2f' % (rtext[i], '.' * dots, R.Q[i])
    print '(rel) bias........%.2f' % (R.Q[-1])


def cross_validate_one(idx):
    global gdata
    (qs, threads) = gdata
    w_dim = qs[0].f.shape[0]
    q_dim = qs[0].r.shape[0]
    R = Relevance(w_dim, q_dim)
    np.random.seed(17151711 + idx * 2 + 1)
    if idx == 0:
        R.train(qs, learning_rate=0.02, nepoch=200, evaluate_loss_after=100,
                batch_size=10, reg=1e-3)
        res = 0
    else:
        np.random.shuffle(qs)
        trainvalborder = len(qs) * (threads - 2) / (threads - 1)
        R.train(qs[:trainvalborder], learning_rate=0.02, nepoch=200, evaluate_loss_after=100,
                batch_size=10, reg=1e-3)
        res = stats(R, qs[trainvalborder:])
        print 'Loss after training on train(idx=%d): %.2f' % (idx, R.calculate_loss(qs[:trainvalborder]))
        print 'Stats after training on test(idx=%d): %.2f' % (idx, res)
    return res, R


def cross_validate_all(qstrain):
    global gdata
    threads = 4
    gdata = (qstrain, threads + 1)
    i = 0
    pool = Pool()
    percs = []
    for res in pool.imap(cross_validate_one, range(threads + 1)):
        perc, R = res
        if i == 0:
            retR = R
            i += 1
        else:
            percs.append(perc)
    pool.close()
    print percs
    print 'mean perc after val =', sum(percs) / threads
    return retR

results = []


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
        results.append((q.qtext, yt))
        if q.y == yt:
            corr += 1
    print '%.2f%% correct (%d/%d)' % (float(corr) / i * 100, corr, i)
    return float(corr) / i * 100


def rewrite_output():
    lines = []
    for line in csv.reader(open(outfile), delimiter='\t', skipinitialspace=True):
        for qtext, yt in results:
            if line[1] == qtext:
                if yt == 1:
                    line[3] = 'YES'
                else:
                    line[3] = 'NO'
        lines.append(line)
    writer = csv.writer(open(outfile, 'wr'), delimiter='\t')
    for line in lines:
        writer.writerow(line)


def query_yes_no(question, default="yes"):
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    prompt = " [y/n] "
    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

if __name__ == '__main__':
    np.random.seed(17151711)
    train()
