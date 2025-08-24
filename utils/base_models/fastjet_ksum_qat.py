# Deepsets models in a format friendly for synthetisation. 
#From https://github.com/fastmachinelearning/l1-jet-id/blob/main/fast_jetclass/deepsets/deepsets_synth.py

import numpy as np
import sys 

import os

import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as KL
from tensorflow.keras import backend as KB

import qkeras
from qkeras import QActivation

from custom_layers import KSum 



def ksum_qat(
    input_size: tuple, 
    phi_layers: list = [],
    rho_layers: list = [],
    output_dim: int = 1,
    activ: str = "relu",
    vector_bits: list = [], #Intermediate and output vectors are quantized to this precision. 
    model_bits: list = [] #model architecture bits - eg. for weights and biases. 
):

    quant = format_quantiser(model_bits) 
    activ = format_qactivation(activ, vector_bits)

    deepsets_input = keras.Input(shape=input_size, name="input_layer") 

    # Phi network.
    x = qkeras.QDense(
        phi_layers[0], kernel_quantizer=quant, bias_quantizer=quant, name=f"phi{1}"
   )(deepsets_input)
    x = qkeras.QActivation(activ)(x)

    for i, layer in enumerate(phi_layers[1:]):
        x = qkeras.QDense(
            layer, kernel_quantizer=quant, bias_quantizer=quant, name=f"phi{i+2}"
        )(x)
        x = qkeras.QActivation(activ)(x)

    # Aggregator
    x = KSum(name="KSum")(x)
    #quantize the output of this too. 
    x = qkeras.QActivation(f"quantized_bits({vector_bits[0]},{vector_bits[1]})")(x)

    # Rho network.
    for i, layer in enumerate(rho_layers):
        x = qkeras.QDense(
            layer, kernel_quantizer=quant, bias_quantizer=quant, name=f"rho{i+1}"
        )(x)
        x = qkeras.QActivation(activ)(x)

    #try this to quantize the model's output... 
    x = KL.Dense(1, name="regression")(x)
    regression_output = qkeras.QActivation(f"quantized_bits({vector_bits[0]},{vector_bits[1]})")(x)

    # Final model 
    deepsetsKsumQat = keras.Model(
        inputs=deepsets_input,
        outputs=regression_output,
        name="dsKsumQat"
    )

    return deepsetsKsumQat



#fastjetclass's function for getting model activations 
def get_model_activations(model: keras.Model):
    """Looks at the layers in a model and returns a list with all the activations.

    This is done such that the precision of the activation functions is set separately
    for the synthesis on the FPGA.
    """
    model_activations = []
    for layer in model.layers:
        if "activation" in layer.name:
            model_activations.append(layer.name)

    return model_activations


def format_quantiser(nbits: list):
    #Format the quantisation of the ml floats in a QKeras way.
    if len(nbits)==1 and nbits[0] == 1:
        return "binary(alpha=1)"
    elif len(nbits)==1 and nbits[0] == 2:
        return "ternary(alpha=1)"
    elif len(nbits)==1: 
        return f"quantized_bits({nbits[0]}, 0, alpha=1)"
    else:
        return f"quantized_bits({nbits[0]}, {nbits[1]}, alpha=1)" 


def format_qactivation(activation: str, nbits: list) -> str:
    """Format the activation function strings in a QKeras friendly way."""
    return f"quantized_{activation}({nbits[0]}, {nbits[1]})"
