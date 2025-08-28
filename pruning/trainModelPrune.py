import tensorflow as tf
 
from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
from tensorflow_model_optimization.sparsity.keras import strip_pruning, PruningSummaries, UpdatePruningStep, prune_scope
from tensorflow.keras.models import load_model
import numpy as np 
from utils import SequenceDS, load_data_into_batch, load_test_data

import pickle
import os
import argparse
import tf_keras as keras

from fastjet_ksum_floats import ds_ksum
import sys 
sys.path.append(os.path.join(os.path.dirname(__file__), '/home/ali2/hls4ds/training_programs/wk4/regression_only/custom_layers'))
from init_layers import KSum 

import tempfile

#function from https://www.tensorflow.org/model_optimization/guide/pruning/comprehensive_guide.md
def get_gzipped_model_size(model):
  # Returns size of gzipped model, in bytes.
  import os
  import zipfile

  _, keras_file = tempfile.mkstemp('.h5')
  model.save(keras_file, include_optimizer=False)

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(keras_file)

  return os.path.getsize(zipped_file)


#potential 2-phase strategy: 
"""Phase 1: Prune the model 
- load pre-trained weights from the 100-epoch trained Keras model  
- do this WITHOUT early stopping to achieve target sparsity 
Phase 2: Fine Tuning 
- train WITHOUT pruning and WITH earlyStopping + restore_best_weights to recover the best accuracy without affecting sparsity"""


if __name__ == '__main__': 

    parser = argparse.ArgumentParser(description='Without option argument, it will not run properly.')
    parser.add_argument('-gpu', '--logical_devices', type = int, required = True, help = "Set visible GPU devices" )
    parser.add_argument('-eps', '--epochs', type=int, default=20, help="Number of pruning epochs.") 
    parser.add_argument('-epb', '--epoch_begin', type=int, default=3, help="Epoch number when pruning schedule begins.")
    parser.add_argument('-pl', '--phi_layers', type=int, default=5, help="Number of phi layers.")
    parser.add_argument('-pn', '--phi_nodes', type=int, default=64, help="Number of phi nodes.")  
    parser.add_argument('-fl', '--F_layers', type=int, default=5, help="Number of F/rho layers.")
    parser.add_argument('-fn', '--F_nodes', type=int, default=64, help="Number of F/rho nodes.") 
    parser.add_argument('-nm', '--name', type=str, required=True, help="Name of model to be saved under.")
    parser.add_argument('-ts', '--tgt_sparsity', type=float, default=0.5, help="Target sparsity.")
    parser.add_argument('-b', '--batch_size', type=int, default=4096, help="Mini batch size.") 
    parser.add_argument('-i', '--data_path', type=str, default="/home/ali2/hls4ds/training_programs/wk4/regression_only/processedDataPadded-6/", help="Path to processed files location.")  
    parser.add_argument('-o', '--save_path', type=str, default="/home/ali2/hls4ds/training_programs/wk4/regression_only/prune/keras_models/", help="Directory where model materials will be saved.")
    parser.add_argument('-w', '--weights_path', type=str, default="/home/ali2/hls4ds/training_programs/wk4/regression_only/PTQ/keras_models/pad-6/ksum_64n_pad-6_training_materials/ksum_64n_pad-6_weights.h5", help="Path to pre-trained weights.") 
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

    phi_nodes = [args.phi_nodes] * args.phi_layers
    F_nodes = [args.F_nodes] * args.F_layers

    f_idx_list, v_idx_list = list(range(1,25)), list(range(25,31)) 
    #f_idx_list, v_idx_list = list(range(1,3)), list(range(3,4))
    f = [args.data_path + f"pre_log10_v128_f{idx:03d}.p" for idx in f_idx_list]
    v = [args.data_path + f"pre_log10_v128_f{idx:03d}.p" for idx in v_idx_list]

    model = ds_ksum(
        input_size=(128,4), 
        phi_layers=phi_nodes,
        rho_layers=F_nodes,
        activ='relu'
        )

    begin_step = int(args.epoch_begin * 2093) #begin after 3 epochs; steps must be cast explicitly to int
    end_step = int(begin_step + args.epochs*2093)

    #https://github.com/fastmachinelearning/hls4ml-tutorial/blob/main/part3_compression.ipynb 
    pruning_params = {"pruning_schedule": pruning_schedule.PolynomialDecay(
        initial_sparsity=0.0, 
        final_sparsity=args.tgt_sparsity, 
        begin_step=begin_step, 
        end_step = end_step, 
        frequency=200) #default is 100 -> going higher = less noise 
        }

    model.load_weights(args.weights_path) #recommended for accuracy by tf: https://www.tensorflow.org/model_optimization/guide/pruning/comprehensive_guide.md 
    print("Pretrained weights loaded.")

    model = prune.prune_low_magnitude(model, **pruning_params)
    model.summary()

    regression_loss = keras.losses.MeanSquaredError()
    losses = regression_loss
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    base_dir = args.save_path
    file_path = os.path.join(base_dir, f"{args.name}_training_materials/")
    os.makedirs(file_path, exist_ok=True)

    log_dir = tempfile.mkdtemp(
        dir=file_path, 
        prefix=f"{args.name}_training_log_"
    )

    # Specify the callback function (to be printed in terminal for monitoring progress)
    # Not using earlyStopping nor restore_best_weights to be true because it defeats the entire purpose of the pruning schedule 
    callbacks = [keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=10, min_lr=0.00001, verbose=1),
                 UpdatePruningStep(), 
                 PruningSummaries(log_dir=log_dir)]              

    # Compile the model and start training
    model.compile(optimizer=optimizer, loss=losses, metrics=['accuracy', 
                                                            tf.keras.metrics.Precision(),
                                                            tf.keras.metrics.AUC(),
                                                            tf.keras.metrics.Recall()])
    
    print("Start Training:")
    history = model.fit(x=SequenceDS(f,args.batch_size),
                        validation_data=SequenceDS(v,args.batch_size),
                        epochs=args.epochs,
                        callbacks=callbacks,
                        use_multiprocessing=True,
                        shuffle=True,
                        verbose=1)
    history = history.history
    print("End Training")

    #strip_pruning: this function removes pruning wrappers, saving the model as a "normal" Keras model 
    save_model=strip_pruning(model)
    print("final model")
    save_model.summary()
    print("\n")
    print("Size of gzipped pruned model without stripping: %.2f bytes" % (get_gzipped_model_size(model)))
    print("Size of gzipped pruned model with stripping: %.2f bytes" % (get_gzipped_model_size(save_model)))

    name = args.name

    try:
        save_model.save(os.path.join(file_path, f"{name}.keras"))
        save_model.save(os.path.join(file_path, f"{name}.h5"))

        # Save model architecture as JSON
        with open(os.path.join(file_path, f"{name}.json"), 'w', encoding='utf-8') as outfile:
            outfile.write(save_model.to_json())

        # Save weights
        weights_file_path = os.path.join(file_path, f"{name}_weights.h5")
        save_model.save_weights(weights_file_path)

        # Save training history
        with open(os.path.join(file_path, f"{name}_history.pickle"), 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"{name} model and files saved to: {file_path}")

    except RuntimeError as e:
        print("Error occurred while saving:", e)

    #test if model can be loaded. 
    #with prune_scope():
    model_l = load_model(os.path.join(file_path, f"{name}.keras"), custom_objects={"KSum": KSum})
    print("Keras model loaded.")

     #Verify if sparsity was achieved
    for layer in model_l.layers:
        if layer.weights:  # non-empty list
            w = layer.weights[0].numpy()
            print(f'% of zeros in layer {layer} = {(np.sum(w == 0) / np.size(w))}')

    

