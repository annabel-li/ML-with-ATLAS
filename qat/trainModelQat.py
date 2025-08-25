import numpy as np
import sys 
import pickle, gzip
import argparse
import os

import tensorflow as tf
import tf_keras as keras
from qkeras.utils import model_save_quantized_weights

try:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:  # __file__ not defined in Jupyter notebooks
    repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(repo_root)
from utils.base_models.fastjet_ksum_qat import ksum_qat

from convertModelQatKsum import convert_ksum_model


def keras_to_hls(  
    phi_nodes: list, 
    F_nodes: list,
    train_mtrls: tuple, 
    valid_mtrls: tuple, 
    v_precision: list, 
    wb_precision: list, 
    model_weight_path: str, #only if not training from scratch <--- from the pre-trained model. 
    tr_from_scratch: bool = False, 
    epochs: int=10,
    test_data_path: str="/Processed_Data/", 
    batch_size: int=4096, 
    model_type: str="model", 
    save_path = "/qat/trained_model_materials/", 
):

    train = True 
    name = f"{phi_nodes[0]}.{v_precision[0]}.{v_precision[1]}.wb_{wb_precision[0]}.{wb_precision[1]}.{model_type}"
    #create the model. 

    #check that the QAT model hasn't already been trained: 
    model_specific_save_path = os.path.join(save_path, f"{name}.model")
    if os.path.exists(model_specific_save_path): 
        print(f"{name} model already done.")
        train = False 

    if train == True:  
        model = ksum_qat(
            input_size=(128,4),
            phi_layers=phi_nodes, 
            rho_layers=F_nodes, 
            vector_bits=v_precision, 
            model_bits=wb_precision
        )
        model.summary()
        
        #load weights from the properly trained float32 model; this improves QAT accuracy and drastically reduces training time. 
        if tr_from_scratch == False: 
            model.load_weights(model_weight_path)
            print("Weights from previously trained model loaded.")
    
        #train the model. 
        #Specify the loss function and optimizer
        regression_loss = keras.losses.MeanSquaredError()
        losses = regression_loss
        optimizer = keras.optimizers.Adam(learning_rate=0.001)

        # Specify the callback function (to be printed in terminal for monitoring progress), 
        # change of learning rate on plateau and early stopping 
        callbacks = [keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5, min_lr=0.000001, verbose=1),
                    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1)]
                    
        # Compile the model and start training
        model.compile(optimizer=optimizer, loss=losses, metrics=['accuracy', 
                                                                tf.keras.metrics.Precision(),
                                                                tf.keras.metrics.AUC(),
                                                                tf.keras.metrics.Recall()])

        
        print(f"Start Training for the {name} model:")
        history = model.fit(x = train_mtrls[0], 
                            y = train_mtrls[1], 
                            validation_data=valid_mtrls, 
                            epochs=epochs,
                            callbacks=callbacks,
                            shuffle=True,
                            verbose=1)
        history = history.history
        print("Training ended.")

        # Save full model (including weights and architecture)
        try:

            model_specific_save_path = os.path.join(save_path, f"{name}.model")
            os.makedirs(model_specific_save_path, exist_ok=True)
            
            model.save(os.path.join(model_specific_save_path, f"{name}.keras"))
            model.save(os.path.join(model_specific_save_path, f"{name}.h5"))

            # Save model architecture as JSON
            with open(os.path.join(model_specific_save_path, f"{name}.json"), 'w', encoding='utf-8') as outfile:
                outfile.write(model.to_json())

            # Save weights
            weights_save_path = os.path.join(model_specific_save_path, f"{name}_weights.h5")
            model.save_weights(weights_save_path)

            model_save_quantized_weights(model, os.path.join(model_specific_save_path, f"{name}_qweights.h5"))

            # Save training history
            with open(os.path.join(model_specific_save_path, f"{name}_history.pickle"), 'wb') as handle:
                pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

            if prune: 

                strip_pruned_model = strip_pruning(model)
                strip_pruned_model.save(os.path.join(model_specific_save_path, f"{name}_strip_pruned.keras"))
                print(f"Stripped model saved to: {os.path.join(model_specific_save_path, f'{name}_strip_pruned.keras')}")

            print(f"Model and files saved to: {model_specific_save_path}")

        except RuntimeError as e:
            print("Error occurred while saving:", e) 

        
        #load/generate test data: 
        inputs, regression_targets = dummy_data_gen(num_samples=1000)
        print("Test data loaded.")

        print("Configuring & converting hls model.")
        convert_ksum_model( 
            model = model, #trained keras model
            num_nodes = phi_nodes[0],  
            inputs=inputs, 
            regression_targets=regression_targets, 
            model_type = model_type, 
            v_precision= v_precision, 
            wb_precision=wb_precision
        )










