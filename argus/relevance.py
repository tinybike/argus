# -*- coding: utf-8 -*-
"""
ML model combining classification and relevance together:
YES/NO probability = sum(relevance*classification)/sum(relevance)

FIXME: Rename this module + class, it's not about just relevance anymore.
"""
import numpy as np


def sig(x):
    return 1. / (1 + np.exp(-x))


sigm = np.vectorize(sig)


class Q:
    def __init__(self, f, r, y):
        self.f = f
        self.r = r
        self.y = y


class Relevance:
    def __init__(self, w_dim, q_dim):
        self.w_dim = w_dim
        self.q_dim = q_dim
        init_mean = 0.01
        # Randomly initialize the network parameters
        self.W = np.random.normal(0., init_mean, w_dim + 1)
        self.Q = np.random.normal(0., init_mean, q_dim + 1)

    def forward_propagation(self, f, r):
        try:
            f = np.vstack((f, np.ones(f.shape[1])))
            r = np.vstack((r, np.ones(r.shape[1])))
        except IndexError:
            f = np.hstack((f, 1))
            r = np.hstack((r, 1))

        u = np.dot(self.W, f)
        v = np.dot(self.Q, r)
        t = sigm(v)
        s = sigm(u)
        yt = np.inner(s, t) / np.sum(t)
        return yt

    def probs_rels(self, f, r):
        try:
            f = np.vstack((f, np.ones(f.shape[1])))
            r = np.vstack((r, np.ones(r.shape[1])))
        except IndexError:
            f = np.hstack((f, 1))
            r = np.hstack((r, 1))
        u = np.dot(self.W, f)
        v = np.dot(self.Q, r)
        t = sigm(v)
        s = sigm(u)
        probs = s
        rels = t / max(t)
        return probs, rels

    def grad(self, f, r, y, reg):
        try:
            f = np.vstack((f, np.ones(f.shape[1])))
            r = np.vstack((r, np.ones(r.shape[1])))
        except IndexError:
            f = np.hstack((f, 1))
            r = np.hstack((r, 1))

        u = np.dot(self.W, f)
        v = np.dot(self.Q, r)
        t = sigm(v)
        s = sigm(u)
        yt = np.inner(s, t) / np.sum(t)
        if y == 1:
            dLdyt = 1. / yt
        else:
            dLdyt = 1. / (yt - 1)

        dytdsum = 1. / np.sum(t)
        dstds = t
        dsdu = s * (1 - s)
        try:
            A = np.diag(dstds * dsdu)
            dsudw = np.dot(f, A)
            dsumdw = np.sum(dsudw, 1)
        except ValueError:
            A = dstds * dsdu
            dsudw = np.dot(f, A)
            dsumdw = np.sum(dsudw)

        dLdW = dLdyt * dytdsum * dsumdw
        #################################
        try:
            B = np.diag(t * (1 - t))
            dtdq = np.dot(r, B)
            I = np.dot(dtdq, s) / np.sum(t)
            II = np.sum(t * s) * np.sum(dtdq, 1) / (np.sum(t) ** 2)
        except ValueError:
            B = t * (1 - t)
            dtdq = np.dot(r, B)
            I = np.dot(dtdq, s) / np.sum(t)
            II = np.sum(t * s) * np.sum(dtdq) / np.sum(t) ** 2

        dytdq = I - II
        dLdQ = dLdyt * dytdq
        dLdQ -= reg * self.Q
        dLdW -= reg * self.W
        return dLdW, dLdQ

    def loss_one(self, f, r, y):
        yt = self.forward_propagation(f, r)
        if y == 1:
            loss = -np.log(yt)
        else:
            loss = -np.log(1 - yt)
        return loss

    def calculate_loss(self, qs):
        loss = 0
        for q in qs:
            loss += self.loss_one(q.f, q.r, q.y)
        return loss / len(qs)

    def save(self, path):
        np.save(path + '/' + 'W.npy', self.W)
        np.save(path + '/' + 'Q.npy', self.Q)

    def load(self, path):
        self.W = np.load(path + '/' + 'W.npy')
        self.Q = np.load(path + '/' + 'Q.npy')
        self.w_dim = np.size(self.W)
        self.q_dim = np.size(self.Q)

    def train(self, qs, learning_rate=0.1, nepoch=500, evaluate_loss_after=5,
              batch_size=10, reg=1e-3, train_rel=[1, 1]):
        """
        :param qs: list of Q objects
        :param learning_rate:
        :param nepoch:
        :param evaluate_loss_after: Print current loss after n epochs.
        :param batch_size:
        :param reg:
        :param train_rel: use np.array mask if you dont want some relevance features to be trained.
        (format [0,1,0,0] only trains second feature)
        :return:
        """
        # We keep track of the losses so we can plot them later
        losses = []
        num_examples_seen = 0
        for epoch in range(nepoch):
            # Optionally evaluate the loss
            if (epoch % evaluate_loss_after == 0):
                loss = self.calculate_loss(qs)
                losses.append((num_examples_seen, loss))
                print "Loss after epoch=%d: %f" % (epoch, loss)
            #                # Adjust the learning rate if loss increases
            #                if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
            #                    learning_rate = learning_rate * 0.5
            #                    print "Setting learning rate to %f" % learning_rate
            dLdW = np.zeros(np.size(self.W))
            dLdQ = np.zeros(np.size(self.Q))
            i = 1
            np.random.shuffle(qs)
            for q in qs:
                dldw, dldq = self.grad(q.f, q.r, q.y, reg)
                dLdW += dldw
                dLdQ += dldq
                if i % batch_size == 0:
                    self.W += dLdW * learning_rate * train_rel[0]
                    self.Q += dLdQ * learning_rate * train_rel[1]

                    dLdW = np.zeros(np.size(self.W))
                    dLdQ = np.zeros(np.size(self.Q))
                i += 1
            self.W += dLdW * learning_rate * train_rel[0]
            self.Q += dLdQ * learning_rate * train_rel[1]


if __name__ == '__main__':
    R = Relevance(2, 2)
    f1 = np.array([3, 1])
    r1 = np.array([-2, 1])
    f2 = np.array([-1, 5])
    r2 = np.array([1, 1])
    f = np.vstack((f1, f2, f1, f2)).T
    r = np.vstack((r1, r2, r1, r2)).T
    qs = [Q(f, r, 1), Q(f1, r1, 0)]

    print R.forward_propagation(f, r)
    print R.forward_propagation(f1, r1)
    #    print R.calculate_loss(qs)


    R.train(qs)
    print R.forward_propagation(f, r)
    print R.forward_propagation(f1, r1)
    #    R.W = np.array([-3, 5])
    #    R.Q = np.array([3, 1])
    #    print R.forward_propagation(f,r)
    #    print R.calculate_loss(qs)
    print R.probs_rels(f, r)

    print R.W
    print R.Q
