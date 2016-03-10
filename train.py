# -*- coding: utf-8 -*-
"""
Training Relevance model happens here, you can change various parameters in train().
"""
import numpy as np
from argus.relevance import Relevance, Q
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Masking, TimeDistributedDense, TimeDistributedMerge
from keras.optimizers import SGD
import csv
import sys
from layer import ClasRel
import keras.preprocessing.sequence as prep
outfile = 'tests/outfile.tsv'
trainIDs = []


def saveIDs():
    ids = np.array(trainIDs)
    np.save('tests/trainIDs/trainIDs.npy', ids)


def extract_xy(qs, max_words):
    f = np.array([prep.pad_sequences(q.f, maxlen=max_words, padding='post',
                                     truncating='post', dtype='float32') for q in qs])
    r = np.array([prep.pad_sequences(q.r, maxlen=max_words, padding='post',
                                     truncating='post', dtype='float32') for q in qs])
    x = np.concatenate((f, r), axis=1)
    x = x.transpose((0, 2, 1))
    y = np.array([q.y for q in qs])
    return x, y


def train():
    max_sentences = 100
    qs_train, qs_test, ctext, rtext = load_features()

    # inverse_features(qstrain, ctext, rtext)
    # inverse_features(qstest)
    # multip_features(qstrain, ctext, rtext)
    # multip_features(qstest)
    zero_features(qs_train, ctext, rtext)
    zero_features(qs_test)
    w_dim = qs_train[0].f.shape[0]
    q_dim = qs_train[0].r.shape[0]

    model = Sequential()
    model.add(ClasRel(w_dim=w_dim, q_dim=q_dim, init='normal', max_sentences=max_sentences,))
    # model.add(TimeDistributedDense(1, input_shape=(max_sentences, w_dim + q_dim)))
    # model.add(TimeDistributedMerge(mode='ave'))

    sgd = SGD(lr=0.02, decay=0., momentum=0., nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd, class_mode='binary')

    print 'compiled'

    x_train, y_train = extract_xy(qs_train, max_sentences)
    x_test, y_test = extract_xy(qs_test, max_sentences)

    model.fit(x_train, y_train, batch_size=10, nb_epoch=200, show_accuracy=True)
    score = model.evaluate(x_test, y_test, batch_size=32, verbose=1, show_accuracy=True)
    y_ = model.predict(x_test)
    # print y_

    # print model.layers[-1].W.eval(), model.layers[-1].w.eval()
    # print model.layers[-1].Q.eval(), model.layers[-1].q.eval()

    print '\n========================\n'
    # list_weights(R, ctext, rtext)
    # print 'W_shape =', R.W.shape
    # print 'Q_shape =', R.Q.shape
    # print '---------------test'
    stats(model, x_train, y_train)
    stats(model, x_test, y_test)
    # print '---------------train'
    # stats(R, qstrain)
    # if query_yes_no('Save model?'):
    #     R.save('sources/models')
    # if query_yes_no('Rewrite output.tsv?'):
    #     rewrite_output()


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


results = []


def stats(model, x, y):
    i = len(y)
    y = np.reshape(y, (i, 1))
    y_ = model.predict(x)
    corr = np.sum(np.abs(y-y_) < 0.5).astype('int')
    print '%.2f%% correct (%d/%d)' % (float(corr) / i * 100, corr, i)
    return float(corr) / i * 100


def rewrite_output():
    lines = []
    for line in csv.reader(open(outfile), delimiter='\t', skipinitialspace=True):
        for qtext, yt, t in results:
            if line[1] == qtext:
                if yt == 1:
                    line[3] = 'YES'
                else:
                    line[3] = 'NO'
                line[11] = str(t)
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
