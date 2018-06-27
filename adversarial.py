import numpy as np 
import tensorflow as tf
from keras import backend as K 
from keras.layers import Dense
from keras.engine.topology import Layer
from keras.layers.merge import concatenate

# polynomial layer - currently unused
class PolyLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.order = output_dim
        self.output_dim = output_dim
        super(PolyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(self.order + 1, 1),
                                      initializer='uniform',
                                      trainable=True)
        super(PolyLayer, self).build(input_shape)  

    def call(self, x):
        basis = K.concatenate([K.pow(x, i) for i in xrange(self.order + 1)])
        return K.dot(basis, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)


# https://github.com/michetonu/gradient_reversal_keras_tf/blob/master/flipGradientTF.py
def _reverse(x, scale = 1):
    # first make this available to tensorflow
    if hasattr(_reverse, 'N'):
        _reverse.N += 1 
    else:
        _reverse.N = 1
    name = 'reverse%i'%_reverse.N 

    @tf.RegisterGradient(name)
    def f(op, g):
        # this is the actual tensorflow op
        return [scale * tf.negative(g)]

    graph = K.get_session().graph 
    with graph.gradient_override_map({'Identity':name}):
        ret = tf.identity(x)

    return ret 

class GradReverseLayer(Layer):
    def __init__(self, scale = 1, **kwargs):
        self.scale = scale
        super(GradReverseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.trainable_weights = []
        super(GradReverseLayer, self).build(input_shape)  

    def call(self, x):
        return _reverse(x, self.scale)

    def get_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {}
        base_config = super(GradReverseLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Adversary(object):
    def __init__(self, n_output_bins, n_outputs=1, scale=1):
        self.scale = scale 
        self.n_output_bins = n_output_bins
        self.n_outputs = n_outputs 
        self._outputs = None 

    def __call__(self, inputs):
        self._reverse = GradReverseLayer(self.scale)(inputs)
        self._dense1  = Dense(20, activation='relu')(self._reverse)
        self._dense2  = Dense(20, activation='relu')(self._dense1)
        self._mu      = Dense(self.n_output_bins, activation="linear" ,name='a') (self._dense2)
        self._sigma   = Dense(self.n_output_bins, activation=K.exp    ,name='b') (self._dense2)
        self._pi      = Dense(self.n_output_bins, activation="softmax",name='c') (self._dense2)
        self._outputs = concatenate([self._mu, self._sigma, self._pi])
        return self._outputs 

