import tensorflow as tf
import tf_keras as keras 
import numpy as np
import sys 
import os 
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '/home/ali2/hls4ds/training_programs/wk4/regression_only/')) #path to plotModel script and model analysis functions
from model_analysis_functions import print_training_history, model_accuracy 
from ksum_materials import KSum 
import utils
#history_path = '/home/ali2/hls4ds/training_programs/wk4/regression_only/prune/keras_models/prune_0.5_training_materials/prune_0.5_history.pickle'
#keys_to_analyze = ['loss']

#print_training_history(history_path, keys_to_analyze)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Without option argument, it will not run properly.')
    parser.add_argument('-gpu', '--logical_devices', type = int, required = True, help = "Set visible GPU devices" )
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

    test_data_path = "/home/ali2/hls4ds/training_programs/wk4/regression_only/processedDataPadded-6/"

    #t = [test_data_path + f"pre_log10_v128_f{idx:03d}.p" for idx in list(range(31,40)) ]
    t = [test_data_path + f"pre_log10_v128_f{idx:03d}.p" for idx in list(range(31,32))]
    test_batch, cluster_energy = utils.load_data_into_batch(t, data_format='xn')
    regression_targets = test_batch[1]['regression'] 

    kmodel = tf.keras.models.load_model("/home/ali2/hls4ds/training_strategies/PTQ/keras_models/pad-6/ksum_64n_pad-6_training_materials/ksum_64n_pad-6.keras", 
        compile=False, 
        custom_objects={'KSum': KSum})
    prune03 = tf.keras.models.load_model("/home/ali2/hls4ds/training_strategies/prune/keras_models/prune_0.3_training_materials/prune_0.3.keras", 
        compile=False, 
        custom_objects={'KSum': KSum})
    prune05 = tf.keras.models.load_model("/home/ali2/hls4ds/training_strategies/prune/keras_models/prune_0.5_training_materials/prune_0.5r2.keras", 
        compile=False, 
        custom_objects={'KSum': KSum})
    prune075 = tf.keras.models.load_model("/home/ali2/hls4ds/training_strategies/prune/keras_models/prune_0.75_training_materials/prune_0.75.keras", 
        compile=False, 
        custom_objects={'KSum': KSum})
    
    save_path = "/home/ali2/hls4ds/training_strategies/prune/results/"
    num_samples = 500

    print("Kmodel predicting.")
    kpred = kmodel.predict(test_batch[0][:num_samples])
    #np.savetxt(os.path.join(save_path, "kpred.txt"), kpred)
    print("Prune0.3 predicting.")
    p03pred = prune03.predict(test_batch[0][:num_samples])
    #np.savetxt(os.path.join(save_path,"p03pred.txt"), p03pred)
    print("Prune0.5 predicting.")
    p05pred = prune05.predict(test_batch[0][:num_samples])
    #np.savetxt(os.path.join(save_path,"p05pred.txt"), p05pred)
    print("Prune0.75 predicting.")
    p075pred = prune075.predict(test_batch[0][:num_samples])
    #np.savetxt(os.path.join(save_path,"p075pred.txt"), p075pred)

    print("Appending predictions to dictionary.")
    predictions = {}
    predictions["Original Keras"] = kpred
    predictions["30% Sparsity"] = p03pred
    predictions["50% Sparsity"] = p05pred
    predictions["75% Sparsity"] = p075pred

    #get rid of the log normalization
    for key, value in predictions.items(): 
        predictions[key] = np.power(10, value * 10).reshape(-1,)
    
    regression_targets = regression_targets[:num_samples] 
    regression_targets = np.power(10, regression_targets * 10).reshape(-1,)

    print("Getting averages.")
    for key, value in predictions.items():
        print(f"Working on {key}.")
        with np.errstate(divide='ignore'):
            avg = np.mean(value/regression_targets)
        print(f"{key} prediction avg: {avg}")

    #print("Generating plot.")
    #model_accuracy(
    #    predictions, 
    #    regression_targets[:num_samples], 
    #    title="Model predictions from truth: pruning"
    #)



