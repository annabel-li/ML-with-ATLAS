
import numpy as np
import pickle, gzip
import argparse
import sys
import os
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras

#tweak sys path to allow the script in its location to recognize utils as a module 
try:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:  # __file__ not defined in Jupyter notebooks
    repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

sys.path.append(repo_root)

from utils.base_models.fastjet_ksum import ds_ksum
from utils.dummy_data_generator import dummy_data_gen 


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Without option argument, it will not run properly.')
    parser.add_argument('-gpu', '--logical_devices', type = int, required = True, help = "Set visible GPU devices" )
    parser.add_argument('-eps', '--epochs', type=int, default=100, help="Number of training epochs.") 
    parser.add_argument('-pl', '--phi_layers', type=int, default=5, help="Number of phi layers.") 
    parser.add_argument('-pn', '--phi_nodes', type=int, default=64, help="Number of phi nodes.")  
    parser.add_argument('-fl', '--F_layers', type=int, default=5, help="Number of F/rho layers.")
    parser.add_argument('-fn', '--F_nodes', type=int, default=64, help="Number of F/rho nodes.")  
    parser.add_argument('-b', '--batch_size', type=int, default=250, help="Mini batch size.") 
    parser.add_argument('-n', '--nsamples', type=int, default=10000, help="Number of input samples.") 
    parser.add_argument('-i', '--data_path', type=str, default="/Processed_Data/", help="Path to processed files location.")  
    parser.add_argument('-o', '--save_path', type=str, default="/ptq/keras_models/", help="Directory where model materials will be saved.")
    parser.add_argument('-nm', '--name', type=str, required=True, help="Name of model to be saved under.") 

    args = parser.parse_args()

    phi_nodes = [args.phi_nodes] * args.phi_layers
    F_nodes = [args.F_nodes] * args.F_layers

    #configure tensorflow GPUS. This makes training significantly faster. 
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
 
    # Defining neural network model architecture
    model = ds_ksum(
        input_size=(128,4), 
        phi_layers=phi_nodes,
        rho_layers=F_nodes,
        activ='relu'
        )
    model.summary()


    # Specify the loss function and optimizer
    regression_loss = keras.losses.MeanSquaredError()
    losses = regression_loss
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    # Specify the callback function (to be printed in terminal for monitoring progress), 
    # change of learning rate on plateau and early stopping 
    callbacks = [keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=10, min_lr=0.00001, verbose=1),
                 keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1)]
    
    model.compile(optimizer=optimizer, loss=losses, metrics=['accuracy', 
                                                            tf.keras.metrics.Precision(),
                                                            tf.keras.metrics.AUC(),
                                                            tf.keras.metrics.Recall()])

    train_data, train_targets = dummy_data_gen(num_samples=args.nsamples) #training data
    valid_data, valid_targets = dummy_data_gen(num_samples=args.nsamples* 0.25) #validation data 

    print("Start Training:")
    history = model.fit(x=train_data,
                        y=train_targets, 
                        validation_data=(valid_data, valid_targets), 
                        epochs=args.epochs,
                        callbacks=callbacks,
                        shuffle=True,
                        verbose=1)
    history = history.history
    print("End Training")
    
    # Save model architecture, trained hyperparameter values and training history
    base_dir = args.save_path
    # Subdirectory for saving model materials
    file_path = os.path.join(base_dir, f"{args.name}_training_materials/")

    # Create the directory if it doesn't exist
    os.makedirs(file_path, exist_ok=True)
    name = args.name 

    try:
        # Save full model (including weights and architecture)
        model.save(os.path.join(file_path, f"{name}.keras"))
        model.save(os.path.join(file_path, f"{name}.h5"))

        # Save model architecture as JSON
        with open(os.path.join(file_path, f"{name}.json"), 'w', encoding='utf-8') as outfile:
            outfile.write(model.to_json())

        # Save weights
        weights_file_path = os.path.join(file_path, f"{name}_weights.h5")
        model.save_weights(weights_file_path)

        # Save training history
        with open(os.path.join(file_path, f"{name}_history.pickle"), 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Model and files saved to: {file_path}")

    except RuntimeError as e:
        print("Error occurred while saving:", e) 

