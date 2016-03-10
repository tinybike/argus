from keras.layers.wrappers import TimeDistributed as td

from keras import activations, initializations
import keras.constraints
import keras.regularizers
from keras.layers.core import MaskedLayer, Layer, TimeDistributedDense, TimeDistributedMerge, Activation
import keras.backend as K
import theano.tensor as T


class ClasRel(MaskedLayer):

    input_ndim = 3

    def __init__(self, w_dim, q_dim, max_sentences=100, output_dim=1, init='glorot_uniform', activation='linear',
                 activation_w='sigmoid', activation_q='sigmoid', weights=None,
                 regularizers=[None]*4, activity_regularizer=None, constraints=[None]*4,
                 input_dim=None, **kwargs):
        self.max_sentences = max_sentences
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

        kwargs['input_shape'] = (self.max_sentences, self.w_dim + self.q_dim,)
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
        input_shape = self.input_shape
        return (input_shape[0], input_shape[1], 1)

    def get_output(self, train=False):
        X = self.get_input(train)
        x = K.reshape(X, (-1, self.input_shape[-1]))
        f = x[:, :self.w_dim]
        r = x[:, self.w_dim:]
        s_ = K.dot(f, self.W)
        t_ = K.dot(r, self.Q)
        mask = K.switch(s_, 1, 0)
        s = self.activation_w(s_ + self.w) * mask
        t = self.activation_q(t_ + self.q) * mask
        s = K.reshape(s, (-1, self.input_shape[1]))
        t = K.reshape(t, (-1, self.input_shape[1]))

        output = self.activation(K.sum(s * t, axis=1) / T.sum(t, axis=-1)) # T.sum(t, axis=1))
        output = K.reshape(output, (-1, 1))
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



def test(X,w_dim,W,Q,q,w):
    x = K.reshape(X, (-1, w_dim+q_dim))
    f = x[:, :w_dim]
    r = x[:, w_dim:]
    s_ = K.dot(f, W)
    t_ = K.dot(r, Q)
    mask = K.switch(s_, 1, 0)
    s = s_ + w * mask
    t = 1#self.activation_q(t_ + self.q) * mask
    s = K.reshape(s, (-1, max_sentences))
    return mask


if __name__ == '__main__':

    from keras.models import Sequential
    import keras
    from keras.layers.core import Masking
    from keras.layers.embeddings import Embedding
    import numpy as np

    np.random.seed(1337)  # for reproducibility
    w_dim = 20+10
    q_dim = 20
    max_sentences = 3
    nb_questions = 8
    x = np.random.randint(0, 10, size=(nb_questions, max_sentences, w_dim + q_dim))
    # x = np.ones_like(x)
    # x[-1,:,-1] = np.zeros_like(x[-1,:,-1])
    mask = np.ones((nb_questions, max_sentences))

    # size=(questions, q_dim+w_dim, max_sentences)
    y = np.random.randint(0, 2, size=(nb_questions,))
    print x, y

    import theano
    theano.config.on_unused_input = 'ignore'
    # theano.config.exception_verbosity = 'high'

    X = T.tensor3('X')
    W = T.vector('W')
    Q = T.vector('Q')
    q = T.scalar('q')
    w = T.scalar('w')

    pokus = theano.function(inputs=[X,W,Q,w,q],
                            outputs=test(X,w_dim,W,Q,q,w),
                            allow_input_downcast=True)

    W = np.random.randint(0,10,size=(w_dim))
    Q = np.random.randint(0,10,size=(q_dim))
    w = np.random.randint(0,10,size=())
    q = np.random.randint(0,10,size=())

    print 'pokus', pokus(x,W,Q,w,q).shape
    model = Sequential()

    model.add(ClasRel(w_dim=w_dim, q_dim=q_dim, max_sentences=max_sentences))
    #model.add(TimeDistributedDense(1, input_shape=(max_sentences, w_dim + q_dim)))
    #model.add(TimeDistributedMerge(mode='ave'))
    # model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='sgd',
                  class_mode='binary')# sample_weight_mode='temporal')

    print 'compiled'
    print(x)
    # print model.predict(x).shape
    model.fit(x, y, batch_size=5, nb_epoch=100, show_accuracy=True)
    print model.predict(x)

