from time import time

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization
from tensorflow.keras.layers import Dropout, Activation, Layer
from tensorflow.keras.layers import Masking, TimeDistributed
from tensorflow.keras import backend as K


#We used this sum layer in our DeepSets model, but called it without masking. 
#Original author: Dan Guest (https://github.com/dguest/flow-network/blob/master/SumLayer.py)

class KSum(Layer):


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        """Build layer (dummy)."""
        del input_shape

    def call(self, x_data, mask=None):
        """Apply the sum to an input.
        """
        if mask is not None:
            x_data = x_data * K.cast(mask, K.dtype(x_data))[:, :, None]
        #else:
        #    x_data = x_data[:, :, None]
        return K.sum(x_data, axis=1)

    def compute_mask(self, inputs, mask):
        """Dummy: this layer support masking but is not able to compute it.
        """
        del inputs, mask

    def get_config(self):
        """Returns layer config (nothing).
        """
        #base_config = super(KSum, self).get_config()
        base_config = super().get_config()
        config = {}
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        """Returns layer output shape.
        """
        return (input_shape[0], input_shape[-1])

    def get_prunable_weights(self):
        """Returns layer weights (nothing).
        """
        return []
