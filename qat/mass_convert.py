import sys
import io
import os
import collections.abc
import json
import contextlib
import argparse, gzip, pickle, awkward

from tqdm import tqdm
import tensorflow as tf

#tweak sys path to allow the script in its location to recognize utils as a module 
try:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:  # __file__ not defined in Jupyter notebooks
    repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(repo_root)

from utils.dummy_data_generator import dummy_data_gen
from trainModelQat import keras_to_hls

#gets the peak memory usage during model conversion and training. 
import resource 
def peak_memory_usage():
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    factor_mb = 1/1024
    return mem * factor_mb

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Without option argument, it will not run properly.')
    parser.add_argument('-gpu', '--logical_devices', type = int, required = True, help = "Set visible GPU devices" )
    parser.add_argument('-n', '--nodes', type = int, required = True, help = "Number of nodes in Phi and F layers." )
    parser.add_argument('-v', '--v_prec', type = int, nargs=2, required = True, help = "Vector precision for QAT + hls4ml conversion." )
    parser.add_argument('-wb', '--wb_prec', type = int, nargs=2, required = True, help = "Weights & Biases precision for QAT + hls4ml conversion." )
    parser.add_argument('-ep', '--epochs', type = int, default=20, help = "Number of epochs for training." )
    parser.add_argument('-n', '--nsamples', type=int, default=10000, help="Number of input samples.")
    parser.add_argument('-modt', '--model_type', type = str, default="model", help = "Model name/characteristic of model.") #I used names like wbSweep or vSweep to indicate sweep type. 
    parser.add_argument('-bs', '--batch_size', type = int, default=4096, help = "Batch size.")

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

    #load/generate training and validation data here: 
    train_data, train_targets = dummy_data_gen(num_samples=args.nsamples)
    valid_data, valid_targets = dummy_data_gen(num_samples=args.nsamples * 0.25)
            
    model_size = args.nodes
    outfile = "mem_log.txt" 

    print(f"\n--- Processing model: Nodes={model_size}, Vector precision=<{args.v_prec[0]},{args.v_prec[1]}>, WB precision=<{args.wb_prec[0]},{args.wb_prec[1]}> ---")
            
    try:
        keras_to_hls(
            phi_nodes=[model_size]*5,
            F_nodes=[model_size]*5,
            train_mtrls=(train_data, train_targets),
            valid_mtrls=(valid_data, valid_targets),
            epochs=args.epochs, # example epoch count
            model_type=args.model_type, 
            v_precision=args.v_prec, 
            wb_precision=args.wb_prec, 
            save_path="/qat/trained_model_materials/", 
            test_data_path="/Processed_Data/", 
            model_weight_path = "/ptq/keras_models/model_weights.h5",
            tr_from_scratch = False
        )
    
        with open(outfile, "a") as f: 
            f.write(f"Peak memory usage for the {model_size} nodes, V precision<{args.v_prec[0]},{args.v_prec[1]}> + WB precision=<{args.wb_prec[0]},{args.wb_prec[1]}> model: {peak_memory_usage()}\n")

    except RuntimeError as e:
        print(f"RuntimeError for Nodes={model_size}, Vector precision=<{args.v_prec[0]},{args.v_prec[1]}>, WB precision=<{args.wb_prec[0]},{args.wb_prec[1]}>: {e}")
        print("!!! Skipping this configuration and continuing to the next.")
        
