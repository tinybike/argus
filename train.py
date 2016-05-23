# -*- coding: utf-8 -*-
"""
Training ClasRel model happens here, you can change various parameters in train().

We use preprocessed question dataset with auxiliary features (tests/f*.tsv),
and besides regenerating the Keras model, we also update the predictions
in that dataset.
"""
import csv
import importlib
import pickle
import sys
import argparse
import scipy.stats as ss
import numpy as np
from keras.optimizers import SGD

import pysts.embedding as emb
from argus.keras_preprocess import config, load_sets, load_and_train, tokenize, Q

trainIDs = []
params = ['dropout=0', 'inp_e_dropout=0', 'pact="tanh"', 'l2reg=0.01']  # can be replaced by script params


def train_and_eval(test_path, rnn_args, save_to_argus=True, model='rnn'):
    qs_train, c_text, r_text = load_features('tests/feature_prints/train/all_features.tsv')
    qs_val, _, _ = load_features('tests/feature_prints/val/all_features.tsv')
    qs_test, _, _ = load_features('tests/feature_prints/test/all_features.tsv')

    zero_features(qs_train, c_text, r_text)
    zero_features(qs_val)
    zero_features(qs_test)

    w_dim = qs_train[0].c.shape[-1]
    q_dim = qs_train[0].r.shape[-1]
    print 'w_dim=', w_dim
    print 'q_dim=', q_dim

    # ==========================================================
    optimizer = 'adam'
    max_sentences = 50

    # ==========================================================
    modelname = model
    module = importlib.import_module('.'+modelname, 'models')
    conf, ps, h = config(module.config, params+rnn_args)

    runid = '%s-%x' % (modelname, h)
    print('RunID: %s  (%s)' % (runid, ps))

    print('GloVe')
    glove = emb.GloVe(N=conf['embdim'])

    print('Dataset')
    # vocab = pickle.load(open('sources/vocab.txt'))
    y, vocab, gr = load_sets(qs_train, max_sentences)
    y, _, grv = load_sets(qs_val, max_sentences, vocab)
    yt, _, grt = load_sets(qs_test, max_sentences, vocab)
    # pickle.dump(vocab, open('sources/vocab.txt', 'wb'))

    model = load_and_train(runid, module.prep_model, conf, glove, vocab, gr, grv, grt,
                           max_sentences, w_dim, q_dim, optimizer, test_path=test_path)

    ###################################

    print('Predict&Eval (best epoch)')
    loss, acc_t = model.evaluate(grt, show_accuracy=True)
    print('Test: loss=', loss, 'acc=', acc_t)
    loss, acc_tr = model.evaluate(gr, show_accuracy=True)
    print('Train: loss=', loss, 'acc=', acc_tr)
    loss, acc_v = model.evaluate(grv, show_accuracy=True)
    print('Val: loss=', loss, 'acc=', acc_v)
    results = (acc_tr, acc_v, acc_t)

    print '\n========================\n'
    # list_weights(R, ctext, rtext)
    # print 'W_shape =', R.W.shape
    # print 'Q_shape =', R.Q.shape
    # print '---------------test'

    # stats(model, x_train, y_train)
    # stats(model, x_test, y_test)

    # print '---------------train'
    # stats(R, qstrain)
    if save_to_argus:
        if query_yes_no('Save model?'):
            model.save_weights('sources/models/full_model.h5', overwrite=True)
        if query_yes_no('Rewrite tests/f*.tsv predictions?'):
            for g, splitname in [(gr, 'train'), (grv, 'val'), (grt, 'test')]:
                res_dict = zip(g['q_texts'], model.predict(g)['score'][:,0])
                rewrite_output(splitname, dict(res_dict))

    return results


def load_features(afname):
    clas_ixs, rel_ixs = [], []
    c_text, r_text = [], []
    S, R, C, QS, GS, Q_text = [], [], [], [], [], []
    GS_ix = 0
    i = 0
    for line in csv.reader(open(afname), delimiter='\t', skipinitialspace=True):
        try:
            line = [s.decode('utf8') for s in line]
        except AttributeError:  # python3 has no .decode()
            pass
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
        S.append(tokenize(line[1].lower()))  # FIXME: glove should not use lower()
        Q_text.append(line[0])
        QS.append(tokenize(line[0].lower()))
        R.append([float(line[ix]) for ix in rel_ixs])
        C.append([float(line[ix]) for ix in clas_ixs])
        GS.append(float(line[GS_ix] == 'YES'))

    # Determine list segments that concern same question
    # (just different evidence)
    q_t = ''
    ixs = []
    for q_text, i in zip(Q_text, range(len(QS))):
        if q_t != q_text:
            ixs.append(i)
            q_t = q_text

    qs = []
    for i, i_ in zip(ixs, ixs[1:]+[len(QS)]):
        qs.append(Q(Q_text[i], QS[i:i_], S[i:i_], np.array(C[i:i_]), np.array(R[i:i_]), GS[i]))

    return qs, c_text, r_text


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


def stats(model, x, y):
    i = len(y)
    y = np.reshape(y, (i, 1))
    y_ = model.predict({'clr_in': x})['clr_out']
    corr = np.sum(np.abs(y-y_) < 0.5).astype('int')
    print '%.2f%% correct (%d/%d)' % (float(corr) / i * 100, corr, i)
    return float(corr) / i * 100


def rewrite_output(splitname, results):
    lines = []
    out_tsv = 'tests/f%s.tsv' % (splitname,)
    header = []
    for line in csv.reader(open(out_tsv), delimiter='\t', skipinitialspace=True):
        if not header:
            header = line
            continue
        if line[1] in results:  # not necessarily in case of no recall
            y = results[line[1]]
            if y > .5:
                line[3] = 'YES'
            else:
                line[3] = 'NO'
            line[4] = str(y)
        lines.append(line)
    writer = csv.writer(open(out_tsv, 'wb'), delimiter='\t')
    writer.writerow(header)
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


def student_distribution_print(fname, r, alpha=0.95, bonferroni=1.):
    if len(r) > 0:
        bar = ss.t.isf((1 - alpha) / bonferroni / 2, len(r) - 1) * np.std(r) / np.sqrt(len(r))
    else:
        bar = np.nan
    print('%s: %f ±%f (%s)' % (fname, np.mean(r), bar, r))
    return bar


def train_full(runs, pars, model='rnn'):
    if runs is None:
        runs = 16
    else:
        runs = int(runs)
    results = []
    for i in range(runs):
        print 'Full training, run #%i out of %i' % (i+1, runs)
        results.append(train_and_eval(None, pars, False, model))

    tr_acc = [tr for tr, v, t in results]
    t_acc = [t for tr, v, t in results]
    v_acc = [v for tr, v, t in results]
    print '===========RESULTS==========='
    student_distribution_print('Train', tr_acc)
    student_distribution_print('Val', v_acc)
    student_distribution_print('Test', t_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test')
    parser.add_argument('--full_runs')
    parser.add_argument('--model')
    parser.add_argument('-full', action='store_true')
    args, rnn_args = parser.parse_known_args()

    model = vars(args)['model']
    if model is None:
        model = 'rnn'

    if vars(args)['full']:
        train_full(vars(args)['full_runs'], rnn_args, model)
    else:
        train_and_eval(vars(args)['test'], rnn_args, model=model)
