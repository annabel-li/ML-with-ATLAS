from ds_utils import DS_Model 
import tensorflow as tf 
import numpy as np 
import sys 
import os 
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '/home/ali2/hls4ds/utils/'))
from utils import load_test_data
sys.path.append(os.path.join(os.path.dirname(__file__), '/home/ali2/hls4ds/utils/deepsetModels/custom_layers'))
from init_layers import KSum 

nodes = 64
weights = np.random.rand(64,nodes) 
bias = np.random.rand(nodes) 
layername='test'
layernum=0
inputs = np.random.rand(4, 64)

from nn_utils import Dense_Layer, KSum_Layer
newLayer = Dense_Layer(nodes=nodes, weights=weights, bias=bias, layername=layername, layernum=layernum)
print(newLayer.wb.to_numpy())
newLayer.wbfrac_bits = 4 
print(newLayer.wb.to_numpy())
newLayer.in_data = inputs 
print(newLayer.in_data.to_numpy()) 
newLayer.vfrac_bits = 5 
print(newLayer.in_data.to_numpy())

KSum = KSum_Layer(layernum=1)
KSum.in_data = inputs 
print("KSum inputs: ", KSum.in_data.bits)
KSum.vfrac_bits = 3 
print("KSum adjusted inputs:", KSum.in_data.to_numpy())


"""
keras_file="/home/ali2/hls4ds/training_scripts/PTQ/keras_models/pad-6/ksum_64n_pad-6_training_materials/ksum_64n_pad-6.keras"
vec_precision = {"n_word": 27, "n_frac": 16}
wb_precision = {"n_word": 18, "n_frac": 15}

testDS = DS_Model(keras_file = keras_file, 
    vec_precision = vec_precision, 
    wb_precision = wb_precision 
)

testDS.print_attr()

print("Quantized concatenated weights & bias: ", testDS.model['phi1'].wb.to_numpy())

data_dict = load_test_data(
    file_path="/home/ali2/hls4ds/old_materials/ignore/regression_only/processedDataPadded-6/", 
    file_range=[31, 32],
    num_samples=1000
) 

inputs = data_dict['Inputs']
targets = data_dict['Regression targets']

kmodel = tf.keras.models.load_model(keras_file, compile=False, custom_objects={"KSum": KSum})
kpred = kmodel.predict(inputs)
qpred = testDS.predict(inputs)
#print(f"Time for APy Python model prediction: {testDS.predict_duration}s.")

print("Sample Keras prediction: ", kpred[:10])
print("Sample quantized DS prediction: ", qpred[:10])
"""

