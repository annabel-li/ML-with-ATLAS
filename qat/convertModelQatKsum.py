"""PLEASE NOTE: 
These imports below are NOT included in my repository as I don't own the rights to the work. 
To convert custom layers with hls4ml, you will have to implement extension API - see the 
README of the ptq folder for more details."""
#from ksum_materials import KSum, HSum, FuseSumActivation, HSumConfigTemplate, HSumFunctionTemplate, register_ksum, parse_sum_layer 


import sys
import hls4ml
import os

import numpy as np
import tensorflow as tf

import tf_keras as keras
from qkeras import QDense, QActivation

import gzip, pickle
from hls4ml.model.profiling import numerical, get_ymodel_keras #needed for tracing

#tweak sys path to allow the script in its location to recognize utils as a module 
try:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:  # __file__ not defined in Jupyter notebooks
    repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(repo_root)
from utils.base_models.fastjet_ksum_qat import get_model_activations


def convert_ksum_model(
    model: keras.Model, 
    num_nodes: int,  
    inputs: np.ndarray, 
    regression_targets: np.ndarray, 
    model_type: str,
    wb_precision: list, 
    v_precision: list
    ): 

    #Register your custom layer with extension API here: 
    #register_ksum()

    keras_model = model 

    #print names of layers 
    model.summary(expand_nested=True)

    name = f"{num_nodes}.{v_precision[0]}.{v_precision[1]}.wb_{wb_precision[0]}.{wb_precision[1]}.{model_type}"
    print(f"Starting conversion pipeline for model {name}.")

    #set up configuration for hls4ml 
    hls4ml_config = hls4ml.utils.config_from_keras_model(
        keras_model,
        granularity="name",
        default_precision=f'ap_fixed<{v_precision[0]},{v_precision[1]}>'
    )

    #set up model activations 
    model_activations = get_model_activations(keras_model)
    hls4ml.model.optimizer.get_optimizer("output_rounding_saturation_mode").configure(
            layers=model_activations, 
            rounding_mode="AP_RND", #rounds activations to the nearest representable value 
            saturation_mode="AP_SAT", #if activation is too big/small for the range, it will be capped at the given max/min value 
        )
    #Note: there are also optimizer passes for quantization and pruning 

    for layer in hls4ml_config['LayerName'].keys():
        hls4ml_config['LayerName'][layer]['Trace'] = True
        #configure precision of model parts; leave the rest (vector-related) as default.
        if 'weight' in hls4ml_config['LayerName'][layer]['Precision']:
            hls4ml_config['LayerName'][layer]['Precision']['weight'] = f'ap_fixed<{wb_precision[0]},{wb_precision[1]}>' 
        
        if 'bias' in hls4ml_config['LayerName'][layer]['Precision']: 
            hls4ml_config['LayerName'][layer]['Precision']['bias'] = f'ap_fixed<{wb_precision[0]},{wb_precision[1]}>'
        
    hls_model = hls4ml.converters.convert_from_keras_model(
        model=keras_model,
        hls_config=hls4ml_config,
        output_dir=f'QAT/ksumQat/hls_outputs/{model_type}_results/{model_type}_example_model/', #this creates an output folder 
        backend='Vitis',
        part="xcvu9p-flga2104-3-e",
        io_type = "io_stream"
    )

    save_path =f"/fast_scratch_2/atlas/mlfpga/ali2/home_overflow_results/qat_tensor_files/qat_{model_type}_files/"
    os.makedirs(save_path, exist_ok=True)

    print("Compiling hls model...")
    hls_model.compile() 
    print("Hls model successfully compiled.")

    print("Evaluating HLS model...")

    hls_pred, hls_trace = hls_model.trace(np.ascontiguousarray(inputs))
    keras_trace = get_ymodel_keras(keras_model, inputs)
    print("Hls model evaluated.")

    #evaluate keras model with test data 
    print("Evaluating Keras model...") 
    keras_pred=keras_model.predict(inputs, verbose=1)
    print("Keras model evaluated.")

    if np.isnan(keras_pred).any() or np.count_nonzero(hls_pred) == 0: #sometimes this occurs at high precisions. 
        with open("error_log.txt", "a") as f: 
            f.write(f"Model.predict for keras model {name} generated NaNs and hls_model.predict() generated zeroes.\n")
            f.write("----------------\n")
        print("Skipping this model and continuing to the next.")

    #Save trace files 
    else: 

        output_dir = os.path.join(save_path + f"{name}.model")
        os.makedirs(output_dir, exist_ok=True)

        path = os.path.join(save_path, "regression_targets.txt")
        np.savetxt(path, regression_targets)

        path = os.path.join(output_dir, f"keras_predictions.txt")
        np.savetxt(path, keras_pred)

        path = os.path.join(output_dir, f"hls_predictions.txt")
        np.savetxt(path, hls_pred)

        #save trace dictionaries
        with open(os.path.join(output_dir, "keras_trace_dict.pkl"), 'wb') as file: 
            pickle.dump(keras_trace, file, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(os.path.join(output_dir, "hls_trace_dict.pkl"), 'wb') as file: 
            pickle.dump(hls_trace, file, protocol=pickle.HIGHEST_PROTOCOL)                

        print(f"Post-hls4ml conversion {name} model materials and results generated and saved to {output_dir}.\n")

            

