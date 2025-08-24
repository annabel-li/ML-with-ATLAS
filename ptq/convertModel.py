"""PLEASE NOTE: 
These imports below are NOT included in my repository as I don't own the rights to the work. 
To convert custom layers with hls4ml, you will have to implement extension API - see the 
README of the ptq folder for more details."""
#from ksum_materials import KSum, HSum, FuseSumActivation, HSumConfigTemplate, HSumFunctionTemplate, register_ksum, parse_sum_layer 

import sys
import io
import os
import collections.abc
import json
import contextlib

import numpy as np
import tensorflow as tf
import hls4ml

from tensorflow import keras 
from tqdm import tqdm

import argparse, gzip, pickle, awkward

from fastjet_ksum import get_model_activations
from hls4ml.model.profiling import numerical, get_ymodel_keras

import matplotlib.pyplot as plt

from utils.dummy_data_generator import dummy_data_gen #import your test data loading function here. 


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Without option argument, it will not run properly.')
    parser.add_argument('-gpu', '--logical_devices', type = int, required = True, help = "Set visible GPU devices" )
    parser.add_argument('-i', '--file_path', type=str, default="/ptq/keras_models/", help="File path to trained keras model.")
    parser.add_argument('-t', '--test_data_path', type=str, default="/Processed_Data/", help="File path to test data.")
    parser.add_argument('-s', '--save_path', type=str, default="/ptq/test_outputs/", help="File path for saving.")
    parser.add_argument('-conv', '--convert_to_hls', choices=['t','f'], help="Enter t to compile. Enter f to use files from previous compilation.")
    parser.add_argument('-tr', '--trace', choices=['t','f'], help="Enter t to trace.")
    parser.add_argument('-n', '--nodes', required=True, help="Number of nodes in model.")
    parser.add_argument('-m', '--hls_model_path', type=str, default='/ptq/hls4ml_models/', help="File path to previous hls model files.")
    parser.add_argument('-nm', '--model_name', type=str, default='model', help="Model name for files to be saved under.")
    parser.add_argument('-vfb', '--v_full_bits', nargs="+", type=int, default=[16], help="Total bitwidth range (inclusive) for intermediate/output vector sweep.")
    parser.add_argument('-vib', '--v_int_bits', nargs="+", type=int, default=[6], help="Integer bitwidth range (inclusive) for intermediate/output vector sweep.")
    parser.add_argument('-wbfb', '--wb_full_bits', nargs="+", type=int, default=[16], help="Total bitwidth range (inclusive) for intermediate/output weights & biases sweep.")
    parser.add_argument('-wbib', '--wb_int_bits', nargs="+", type=int, default=[6], help="Integer bitwidth range (inclusive) for intermediate/output weights & biases sweep.")
    args = parser.parse_args()

    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[args.logical_devices], True)
    print(gpus)
    try:
        tf.config.set_visible_devices(gpus[args.logical_devices], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

    #Generate test data
    inputs, regression_targets = dummy_data_gen(num_samples=1000)

    #save regression targets for comparison. 
    path = (os.path.join(args.save_path, "regression_targets.txt"))
    np.savetxt(path, regression_targets)

    #Here, you would register your custom layer with extension API: 
    #register_ksum()

    #load the model 
    model=tf.keras.models.load_model(args.file_path, compile=False, custom_objects={'KSum': KSum}) 

    #print names of layers 
    model.summary(expand_nested=True)

    nodes = args.nodes
    hls_folder_name = f"hls_{args.model_name}"

    #create lists based on the specified range for the design sweep. 
    v_int_range = np.arange(args.v_int_bits[0], args.v_int_bits[1] + 1) 
    v_full_range = np.arange(args.v_full_bits[0], args.v_full_bits[1] + 1) 
    wb_int_range = np.arange(args.wb_int_bits[0], args.wb_int_bits[1] + 1) 
    wb_full_range = np.arange(args.wb_full_bits[0], args.wb_full_bits[1] + 1)   

    #4-way for loop. 
    for v_full_bits in v_full_range: 
        for v_int_bits in v_int_range: 
            for wb_full_bits in wb_full_range: 
                for wb_int_bits in wb_int_range: 
                    try: 

                        precision = [v_full_bits, v_int_bits]
                        if v_int_bits > v_full_bits:  
                            print(f"Skipping vector int bits = {v_int_bits} and full bits = {v_full_bits}")
                            continue
                        if wb_int_bits > wb_full_bits: 
                            print(f"Skipping weights & biases int bits = {wb_int_bits} and full bits = {wb_full_bits}")

                        name = f"{nodes}.{precision[0]}.{precision[1]}.wb_{wb_full_bits}.{wb_int_bits}"
                        #set up configuration for hls4ml. Here, vector and intermediate output precision is the default; 
                        # we only specify precision for the weights and biases.  
                        hls4ml_config = hls4ml.utils.config_from_keras_model(
                            model,
                            granularity="name",
                            default_precision = f'ap_fixed<{precision[0]},{precision[1]}>'
                        )

                        if args.convert_to_hls == 't': 
                        #set up model activations 
                            model_activations = get_model_activations(model)
                        #Use model activations to set optimizer passes, aka method(s) to transform the activations for better hardware implementation 
                        #specifically this function configures how the activations from the NN are treated during conversion 
                            hls4ml.model.optimizer.get_optimizer("output_rounding_saturation_mode").configure(
                                    layers=model_activations, 
                                    rounding_mode="AP_RND", 
                                    saturation_mode="AP_SAT", 
                                )
                            #Note: there are also optimizer passes for quantization and pruning 

                            #convert model & set up tracing 
                            for layer in hls4ml_config['LayerName'].keys():
                                # Enable tracing for each layer
                                hls4ml_config['LayerName'][layer]['Trace'] = True
                                if 'weight' in hls4ml_config['LayerName'][layer]['Precision']:
                                    hls4ml_config['LayerName'][layer]['Precision']['weight'] = f'ap_fixed<{wb_full_bits},{wb_int_bits}>'              
                                if 'bias' in hls4ml_config['LayerName'][layer]['Precision']: 
                                    hls4ml_config['LayerName'][layer]['Precision']['bias'] = f'ap_fixed<{wb_full_bits},{wb_int_bits}>'

                            hls_model = hls4ml.converters.convert_from_keras_model(
                                model=model,
                                hls_config=hls4ml_config,
                                output_dir=f'PTQ/hls_outputs/{hls_folder_name}/{hls_folder_name}_{nodes}n.example.model/', #this creates an output folder 
                                backend='Vitis',
                                part="xcvu9p-flga2104-3-e",
                                io_type = "io_stream"
                            )
                            
                        else:       
                            # Load existing project
                            hls_model = hls4ml.converters.link_existing_project(args.hls_model_path)    

                        #compile hls4ml model and evaluate
                        hls_model.compile() 
                        print("Hls model successfully compiled.")

                        print("Evaluating HLS model...")

                        #enable tracing. 
                        if args.trace == 't': 

                            hls_pred, hls_trace = hls_model.trace(np.ascontiguousarray(inputs))
                            keras_trace = get_ymodel_keras(model, inputs)
                        
                        else: 
                            hls_pred = hls_model.predict(np.ascontiguousarray(inputs))

                        #evaluate keras model with test data 
                        print("Evaluating Keras model...")
                        #inputs = np.expand_dims(inputs, axis=0) #expands the inputs along the third dimension because the models expect 3D inputs - do this if passing 1 sample only. 
                        keras_pred=model.predict(inputs, verbose=1)
                        print("Keras model evaluated.")

                        # True labels
                        regression_targets = regression_targets.reshape(-1,1) #convert to same shape as the model.predict() output arrays 

                        save_path = args.save_path
                        output_dir = os.path.join(save_path + f"{name}.model")
                        os.makedirs(output_dir, exist_ok=True)

                        path = os.path.join(save_path, "regression_targets.txt")
                        np.savetxt(path, regression_targets)

                        path = os.path.join(output_dir, f"keras_predictions.txt")
                        np.savetxt(path, keras_pred)

                        path = os.path.join(output_dir, f"hls_predictions.txt")
                        np.savetxt(path, hls_pred)

                        #Debugging & plotting with tracing 
                        if args.trace == 't': 

                            #with manual configuration, an extra entry is created in the dictionary that needs to be removed
                            #del hls_trace["phi1_linear"]

                            print("keras dict keys:", keras_trace.keys())
                            print("hls dict keys:", hls_trace.keys())

                            #save trace dictionaries
                            with open(os.path.join(output_dir, "keras_trace_dict.pkl"), 'wb') as file: 
                                pickle.dump(keras_trace, file, protocol=pickle.HIGHEST_PROTOCOL)
                                print("Keras trace saved.")
                            
                            with open(os.path.join(output_dir, "hls_trace_dict.pkl"), 'wb') as file: 
                                pickle.dump(hls_trace, file, protocol=pickle.HIGHEST_PROTOCOL)   
                                print("Hls trace saved.")             

                            print(f"Model {name} materials and results generated and saved to {output_dir}.\n")
                        
                    except Exception as e:   
                        print(f"Failed at precision {precision}: {e}")
                        continue


        
    





