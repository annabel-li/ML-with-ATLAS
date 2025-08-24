# Deepsets models in a format friendly for synthetisation. 
#From https://github.com/fastmachinelearning/l1-jet-id/blob/main/fast_jetclass/deepsets/deepsets_synth.py
#Regression only. 

import numpy as np
import sys 
import os

import tensorflow as tf
from tensorflow import keras 
import tensorflow.keras.layers as KL

from utils.base_models.custom_layers import KSum 


def ds_ksum(
    input_size: tuple, 
    phi_layers: list = [],
    rho_layers: list = [],
    output_dim: int = 1,
    activ: str = "relu"
):
    
    deepsets_input = keras.Input(shape=input_size, name="input_layer") 

    # Phi network.
    x = KL.Dense(
        phi_layers[0], name=f"phi{1}"
   )(deepsets_input)
    x = KL.Activation(activ)(x)

    for i, layer in enumerate(phi_layers[1:]):
        x = KL.Dense(
            layer, name=f"phi{i+2}"
        )(x)
        x = KL.Activation(activ)(x)

    # KSum Aggregator
    x = KSum(name="KSum")(x)

    # Rho network.
    for i, layer in enumerate(rho_layers):
        x = KL.Dense(
            layer, name=f"rho{i+1}"
        )(x)
        x = KL.Activation(activ)(x)

    regression_output = KL.Dense(1, name="regression")(x)  

    # Final model with regression output
    deepsets = keras.Model(
        inputs=deepsets_input,
        outputs=regression_output,
        name="dsksum"
    )

    return deepsets


#fastjetclass's function for getting model activations 
def get_model_activations(model: keras.Model):
    """Looks at the layers in a model and returns a list with all the activations.

    This is done such that the precision of the activation functions is set separately
    for the synthesis on the FPGA.
    """
    model_activations = []
    for layer in model.layers:
        if "relu" in layer.name:
            model_activations.append(layer.name)

    return model_activations

