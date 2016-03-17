# -*- coding: utf-8 -*-
"""
Training Relevance model happens here, you can change various parameters in train().
"""
import numpy as np
from keras.models import Sequential, Graph
from keras.layers.core import Activation
import keras.backend as K
from keras.layers.embeddings import Embedding
from keras.layers.core import Masking, TimeDistributedDense, TimeDistributedMerge
from keras.optimizers import SGD
import csv
import sys
from layer import ClasRel
import keras.preprocessing.sequence as prep
from keras.regularizers import l1, l2
import pysts.embedding as emb
from keras_preprocess import config, load_sets, train_and_eval, tokenize, Q
import importlib
import pickle


outfile = 'tests/feature_prints/all_features.tsv'
trainIDs = []


def train():
    qs_train, qs_test, ctext, rtext = load_features()
    # pickle.dump((qs_train, qs_test, ctext, rtext), open('qs.pkl', 'wb'))
    # qs_train, qs_test, ctext, rtext = pickle.load(open('qs.pkl'))

    zero_features(qs_train, ctext, rtext)
    zero_features(qs_test)

    w_dim = qs_train[0].c.shape[-1]
    q_dim = qs_train[0].r.shape[-1]

    # ==========================================================
    epochs = 200
    optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    max_sentences = 50

    # ==========================================================
    modelname = 'rnn'
    params = []
    module = importlib.import_module('.'+modelname, 'models')
    conf, ps, h = config(module.config, params, epochs)

    runid = '%s-%x' % (modelname, h)
    print('RunID: %s  (%s)' % (runid, ps))

    print('GloVe')
    glove = emb.GloVe(N=conf['embdim'])

    print('Dataset')
    y, vocab, gr = load_sets(qs_train, max_sentences)
    yt, _, grt = load_sets(qs_test,max_sentences, vocab)
    pickle.dump(vocab, open('vocab.txt', 'wb'))

    train_and_eval(runid, module.prep_model, conf, glove, vocab, gr, grt,
                   max_sentences, w_dim, q_dim, optimizer)

    ###################################

    print '\n========================\n'
    # list_weights(R, ctext, rtext)
    # print 'W_shape =', R.W.shape
    # print 'Q_shape =', R.Q.shape
    # print '---------------test'

    # stats(model, x_train, y_train)
    # stats(model, x_test, y_test)

    # print '---------------train'
    # stats(R, qstrain)
    # if query_yes_no('Save model?'):
    #     R.save('sources/models')
    # if query_yes_no('Rewrite output.tsv?'):
    #     rewrite_output()


def load_features():
    clas_ixs, rel_ixs = [], []
    c_text, r_text = [], []
    S, R, C, QS, GS, Q_text = [], [], [], [], [], []
    GS_ix = 0
    i = 0
    for line in csv.reader(open(outfile), delimiter='\t', skipinitialspace=True):
        if i == 0:
            i += 1
            for field in line:
                if field == 'Class_GS':
                    GS_ix = line.index(field)
                if '#' in field:
                    clas_ixs.append(line.index(field))
                    c_text.append(field)
                if '@' in field:
                    rel_ixs.append(line.index(field))
                    r_text.append(field)
        if line[0] == 'Question':
            continue
        S.append(tokenize(line[1]))
        Q_text.append(line[0])
        QS.append(tokenize(line[0]))
        R.append([float(line[ix]) for ix in rel_ixs])
        C.append([float(line[ix]) for ix in clas_ixs])
        GS.append(float(line[GS_ix] == 'YES'))
    qs_train = []
    qs_test = []

    q_t = ''
    ixs = []
    for q_text, i in zip(Q_text, range(len(QS))):
        if q_t != q_text:
            ixs.append(i)
            q_t = q_text
    for i, i_, j in zip(ixs, ixs[1:]+[len(QS)], range(len(ixs))):
        if split(Q_text[i]):
            qs_train.append(Q(Q_text[i], QS[i:i_], S[i:i_], np.array(C[i:i_]), np.array(R[i:i_]), GS[i]))
        else:
            qs_test.append(Q(Q_text[i], QS[i:i_], S[i:i_], np.array(C[i:i_]), np.array(R[i:i_]), GS[i]))

    np.save('tests/trainIDs/trainIDs.npy', np.array(trainIDs))
    return qs_train, qs_test, c_text, r_text

trainIDs = list(np.load('tests/trainIDs/trainIDs.npy'))
def split(text):
    if text in trainIDs:
        trainIDs.remove(text)
        return True
    return False

#
# def split(i):
#     return i % 4 < 2


def zero_features(qs, ctext=None, rtext=None):
    """ extend FV with <feature>==0 metafeatures """
    k = 0
    for q in qs:
        flen = q.c.shape[-1]
        for i in range(flen):
            newf = q.c[:, i] == 0
            q.c = np.column_stack((q.c, newf.astype(float)))
            if k == 0 and ctext is not None:
                ctext.append(ctext[i] + '==0')
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
    y_ = model.predict({'clr_in': x})['clr_out']
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