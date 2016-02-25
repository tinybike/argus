#!/usr/bin/python3
"""
Train a KeraSTS model on the Answer Sentence Selection task.

Usage: tools/anssel_train.py MODEL TRAINDATA VALDATA [PARAM=VALUE]...

Example: tools/anssel_train.py cnn data/anssel/wang/train-all.csv data/anssel/wang/dev.csv inp_e_dropout=1/2

This applies the given text similarity model to the anssel task.
Extra input pre-processing is done:
Rather than relying on the hack of using the word overlap counts as additional
features for final classification, individual tokens are annotated by overlap
features and that's passed to the model along with the embeddings.

Final comparison of summary embeddings is by default performed by
a multi-layered perceptron with elementwise products and sums as the input,
while the Ranknet loss function is used as an objective.  You may also try
e.g. dot-product (non-normalized cosine similarity) and binary crossentropy
or ranksvm as loss function, but it doesn't seem to make a lot of difference.

Prerequisites:
    * Get glove.6B.300d.txt from http://nlp.stanford.edu/projects/glove/
"""

from __future__ import print_function
from __future__ import division

import importlib
import sys
import csv

from keras.callbacks import ModelCheckpoint
from keras.layers.core import Activation, Dropout
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.models import Graph

import pysts.embedding as emb
import pysts.eval as ev
import pysts.loader as loader
import pysts.nlp as nlp
from pysts.hyperparam import hash_params
from pysts.vocab import Vocabulary

from pysts.kerasts import graph_input_anssel
import pysts.kerasts.blocks as B
from pysts.kerasts.callbacks import AnsSelCB
from pysts.kerasts.objectives import ranknet, ranksvm, cicerons_1504



s0pad = 60
s1pad = 60


def load_set(fname, vocab=None):
    s0, s1, y, t = loader.load_anssel(fname, skip_oneclass=False)
    # s0=questions, s1=answers

    if vocab is None:
        vocab = Vocabulary(s0 + s1)

    si0 = vocab.vectorize(s0, spad=s0pad)
    si1 = vocab.vectorize(s1, spad=s1pad)
    f0, f1 = nlp.sentence_flags(s0, s1, s0pad, s1pad)
    gr = graph_input_anssel(si0, si1, y, f0, f1)

    return (s0, s1, y, vocab, gr)


def load_sent(q, a, vocab=None):
    s0, s1, y = [q.split(' ')], [a.split(' ')], 1
    # s0=questions, s1=answers

    if vocab is None:
        vocab = Vocabulary(s0 + s1)

    si0 = vocab.vectorize(s0, spad=s0pad)
    si1 = vocab.vectorize(s1, spad=s1pad)
    f0, f1 = nlp.sentence_flags(s0, s1, s0pad, s1pad)
    gr = graph_input_anssel(si0, si1, y, f0, f1)

    return gr


def config(module_config, params):
    c = dict()
    c['embdim'] = 50
    c['inp_e_dropout'] = 1/2
    c['e_add_flags'] = True

    c['ptscorer'] = B.mlp_ptscorer
    c['mlpsum'] = 'sum'
    c['Ddim'] = 1

    c['loss'] = 'binary_crossentropy'
    c['nb_epoch'] = 2
    module_config(c)

    for p in params:
        k, v = p.split('=')
        c[k] = eval(v)

    ps, h = hash_params(c)
    return c, ps, h


def prep_model(glove, vocab, module_prep_model, c, oact, s0pad, s1pad):
    # Input embedding and encoding
    model = Graph()
    N = B.embedding(model, glove, vocab, s0pad, s1pad, c['inp_e_dropout'], add_flags=c['e_add_flags'])

    # Sentence-aggregate embeddings
    final_outputs = module_prep_model(model, N, s0pad, s1pad, c)

    # Measurement

    if c['ptscorer'] == '1':
        # special scoring mode just based on the answer
        # (assuming that the question match is carried over to the answer
        # via attention or another mechanism)
        ptscorer = B.cat_ptscorer
        final_outputs = final_outputs[1]
    else:
        ptscorer = c['ptscorer']

    kwargs = dict()
    if ptscorer == B.mlp_ptscorer:
        kwargs['sum_mode'] = c['mlpsum']
    model.add_node(name='scoreS', input=ptscorer(model, final_outputs, c['Ddim'], N, c['l2reg'], **kwargs),
                   layer=Activation(oact))
    model.add_output(name='score', input='scoreS')
    return model


def build_model(glove, vocab, module_prep_model, c, s0pad=s0pad, s1pad=s1pad):
    if c['loss'] == 'binary_crossentropy':
        oact = 'sigmoid'
    else:
        # ranking losses require wide output domain
        oact = 'linear'

    model = prep_model(glove, vocab, module_prep_model, c, oact, s0pad, s1pad)
    model.compile(loss={'score': c['loss']}, optimizer='adam')
    return model


def eval_questions(sq, sa, labels, results, text):
    question = ''
    label = 1
    avg = 0
    avg_all = 0
    q_num = 0
    correct = 0
    n = 0
    f = open('printout_'+text+'.csv', 'wb')
    w = csv.writer(f, delimiter=',')
    for q, y, t, a in zip(sq, labels, results, sa):
        if q == question:
            n += 1
            avg = n/(n+1)*avg+t/(n+1)
            row = [q, y, t, '', a]
            w.writerow(row)
        else:
            row = [q, y, t, avg, a]
            w.writerow(row)
            if q_num != 0 and abs(label-avg) < 0.5:
                correct += 1
            question = q
            label = y
            avg = t
            q_num += 1
            n = 0
    if q_num != 0 and abs(label-avg) < 0.5:
        correct += 1

    print('precision on separate questions ('+text+'):', correct/q_num)

import pickle
if __name__ == "__main__":
    # modelname, trainf, valf = sys.argv[1:4]
    modelname, trainf, valf = 'rnn', 'data/hypev/argus/argus_train.csv', 'data/hypev/argus/argus_test.csv'
    params = sys.argv[4:]

    module = importlib.import_module('.'+modelname, 'models')
    conf, ps, h = config(module.config, params)

    print('GloVe')
    glove = emb.GloVe(N=conf['embdim'])

    print('Dataset')
# [u'Will', u'the', u'New', u'England', u'Patriots', u'qualify', u'for', u'the', u'Super', u'Bowl?']
# [u'Why', u'a', u'Seattle', u'Seahawks', u'Super', u'Bowl', u'win', u'would', u'be', u'a', u'win', u'for', u'weed']

    # q = 'Will Tomas Berdych win the Wimbledon 2014?'
    # a = 'Andy Murray ready for US Open with volume up on and off court'

    q = 'Will the New England Patriots qualify for the Super Bowl?'
    a = 'Why a Seattle Seahawks Super Bowl win would be a win for weed'
    vocab = pickle.load(open('vocab'))
    gr = load_sent(q, a, vocab)

    print('Model')
    model = build_model(glove, vocab, module.prep_model, conf)
    model.load_weights('sources/models/rnn.h5')
    print('Predict')
    prediction = model.predict(gr)['score'][:,0][0]
    print('PREDICTION', prediction)



