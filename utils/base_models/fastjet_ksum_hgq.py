import numpy as np
import sys 
import os

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as KL

sys.path.append(os.path.join(os.path.dirname(__file__), 'custom_layers'))
from init_layers import KSum 

import HGQ
from HGQ.layers import HQuantize, HDense, HActivation
from qkeras import quantized_bits as q 
from keras.saving import register_keras_serializable
import keras.backend as K

#KSum but compatible with HGQ layers. Similar layer implemented here called HAdd: https://github.com/calad0i/HGQ/blob/master/src/HGQ/layers/misc.py 
#The KSum layer is based on work by Dan Guest. 
@register_keras_serializable(package="HGQ")
class HGQSum(HGQ.layers.base.HLayerBase): 

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        self.post_build(input_shape)

    def forward(self, x_data, training=None, record_minmax=None): #this is where the model's "function" is actually implemented. 
        
        input_bw = self.input_bw
        #tf.print(f"Input bw for {self.name}: {self.input_bw}") 

        summed_data = K.sum(x_data, axis=1)
        return self.paq(summed_data, training=training, record_minmax=record_minmax)

    def get_config(self):
        """Returns layer config (nothing)."""
        #base_config = super(KSum, self).get_config()
        base_config = super().get_config()
        config = {}
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        #Returns layer output shape.
        return (input_shape[0], input_shape[-1])

    def get_prunable_weights(self): #returns nothing (no layer weights)
        return []





def ksum_hgq(
    input_size: tuple, 
    phi_node_list: list,
    rho_node_list: list,
    output_dim: int = 1,
    activ: str = "relu", 
    beta: int=3e-6 #The higher the beta, the more aggressive the quantization will be. 
): 

    deepsets_input = keras.Input(shape=input_size, name="input_layer")

    #first layer in an HGQ model must be quantized. 
    x = HQuantize(input_shape=input_size, name="input_quantizer", beta=beta)(deepsets_input)

    for i, layer in enumerate(phi_node_list): 
        x = HDense(layer, name=f"phi{i}", beta=beta)(x)
        x = HActivation(activation=activ, beta=beta)(x)
        
    x = HGQSum(name="HGQSum")(x)

    # Rho network.
    for i, layer in enumerate(rho_node_list):
        x = HDense(layer, name=f"rho{i}", beta=beta)(x)
        x = HActivation(activation=activ, beta=beta)(x)

    regression_output = HDense(1, name="regression", beta=beta)(x)  

    # Final model with two outputs
    deepsets = keras.Model(
        #inputs=[deepsets_input, inputs_ncells], 
        inputs=deepsets_input,
        outputs=regression_output,
        name="ds_ksum_hgq"
    )

    return deepsets



#to debug why loss is negative during training - see if the issue is with the custom layer. 
def test_model(
    input_size: tuple, 
    beta: int=3e-6, 
    nodes: int=64, 
    layers: int=5
): 

    inputs = keras.Input(shape=input_size, name="input_layer")
    x = HQuantize(input_shape=input_size, name="input_quantizer", beta=beta)(inputs)

    for i in range(layers):
        x = HDense(nodes, name=f"layer_{i}", beta=beta)(x)
        x = HActivation(activation="relu", beta=beta)(x)
    
    regression_output = HDense(1, name="regression", beta=beta)(x)  

    # Final model with two outputs
    model = keras.Model(
        #inputs=[deepsets_input, inputs_ncells], 
        inputs=inputs,
        outputs=regression_output,
        name="test_model"
    )

    return model
