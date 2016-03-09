from keras.layers.wrappers import TimeDistributed as td

from keras import activations, initializations
import keras.constraints
import keras.regularizers
from keras.layers.core import Layer
import keras.backend as K
import theano.tensor as T

max_sentences = 100
class ClasRel(Layer):

    input_ndim = 3

    def __init__(self, w_dim, q_dim, output_dim=1, init='glorot_uniform', activation='linear',
                 activation_w='sigmoid', activation_q='sigmoid', weights=None,
                 regularizers=[None]*4, activity_regularizer=None, constraints=[None]*4,
                 input_dim=None, **kwargs):
        self.max_words = 100
        self.w_dim = w_dim
        self.q_dim = q_dim
        self.input_dim = self.w_dim + self.q_dim
        self.activation = activations.get(activation)
        self.activation_w = activations.get(activation_w)
        self.activation_q = activations.get(activation_q)

        self.init = initializations.get(init)
        self.output_dim = output_dim

        self.W_regularizer = keras.regularizers.get(regularizers[0])
        self.w_regularizer = keras.regularizers.get(regularizers[1])
        self.Q_regularizer = keras.regularizers.get(regularizers[2])
        self.q_regularizer = keras.regularizers.get(regularizers[3])
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)

        self.W_constraint = keras.constraints.get(constraints[0])
        self.w_constraint = keras.constraints.get(constraints[1])
        self.Q_constraint = keras.constraints.get(constraints[2])
        self.q_constraint = keras.constraints.get(constraints[3])
        self.constraints = [self.W_constraint, self.w_constraint,
                            self.Q_constraint, self.q_constraint]

        self.initial_weights = weights

        kwargs['input_shape'] = (self.w_dim + self.q_dim, self.max_words,)
        super(ClasRel, self).__init__(**kwargs)

    def build(self):

        self.W = self.init((self.w_dim, ), name='{}_W'.format(self.name))
        self.w = self.init((), name='{}_w'.format(self.name))
        self.Q = self.init((self.q_dim,), name='{}_Q'.format(self.name))
        self.q = self.init((), name='{}_q'.format(self.name))

        self.trainable_weights = [self.W, self.w, self.Q, self.q]

        self.regularizers = self.fill_regulizers()

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    @property
    def output_shape(self):
        return () #  TODO: figure out whats this

    def get_output(self, train=False):
        X = self.get_input(train)
        f = X[:, :self.w_dim]
        r = X[:, self.w_dim:]
        s = self.activation_w(T.dot(self.W, f) + self.w)
        t = self.activation_q(T.dot(self.Q, r) + self.q)
        # print 'X=', X.type, 'f=', f.type, 'r=', r.type, 's=', s.type, 't=', t.type
        output = self.activation(T.sum(s * t, axis=1) / T.sum(t, axis=1))
        output = T.reshape(output, (-1, 1))
        return output

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(ClasRel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def fill_regulizers(self):
        regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            regularizers.append(self.W_regularizer)

        if self.w_regularizer:
            self.w_regularizer.set_param(self.w)
            regularizers.append(self.w_regularizer)

        if self.Q_regularizer:
            self.Q_regularizer.set_param(self.Q)
            regularizers.append(self.Q_regularizer)

        if self.q_regularizer:
            self.q_regularizer.set_param(self.q)
            regularizers.append(self.q_regularizer)

        return regularizers


from keras.models import Sequential
import keras


from keras.layers.core import Dense, Activation


import numpy as np

np.random.seed(1337)  # for reproducibility

x = np.random.normal(size=(10, 13, 100))
# size=(questions, q_dim+w_dim, J)
y = np.random.randint(0, 2, size=(10,))
print y.shape, y

import theano
theano.config.on_unused_input = 'warn'

model = Sequential()

model.add(ClasRel(w_dim=6, q_dim=7))
# model.add(Activation("relu"))
# model.add(Dense(output_dim=10, init="glorot_uniform"))
# model.add(Activation("softmax"))

model.compile(loss='binary_crossentropy', optimizer='sgd')

print 'compiled'

# X = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)

# size=(output_size, max_words, ??)
print model.predict(x)

model.fit(x, y, batch_size=5, nb_epoch=10)
print model.predict(x)

