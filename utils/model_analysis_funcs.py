"""This file contains functions I made for analyzing discrepancies between Keras and hls4ml models. 
Performance between different hls4ml parameterizations is evaluated with metrics 
such as Mean Percent Error (MPE) and Mean Absolute Error (MAE), though the functions
can be easily edited with metrics such as predicted / true. Many of them also rely on specific folder and path naming 
conventions I developed and setting trace to True during the hls4ml conversion - see the PTQ and QAT folders 
for more details on how to implement this."""

import numpy as np
import matplotlib as plt 
import os 
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import pickle
import tensorflow as tf 


#Function: compares the performance of QAT and PTQ models from their keras counterpart at varying fixed-point precisions and nodes, where one of the precision
# parameters, either integer or decimal, is kept fixed. (1D precision sweep)
def qat_vs_ptq(
    qat_path: str, #path to QAT results
    ptq_path: str, #path to PTQ results
    qat_model_sizes: list, 
    ptq_model_sizes: list, 
    save_path: str, 
    precision_range: list, #last element is exclusive. This contains the range of values you are sweeping through. 
    set_bit: int, #value of the unchanging bit. 
    bit_to_set: int, #0 refers to the M in ap_fixed<M, N> and 1 refers to N ---> eg. if bit_to_set = 1, then you are sweeping through ap_fixed<X, set_bit>
    name: str, #name of the sweep series. 
    num_samples: int=1000, 
    plt_title: str="PTQ vs. QAT MPE from equivalent Keras model", 
    xlabel: str="Fixed-Point Precision", 
    graph_type: str="mpe", #or mae. 
    colours: list=["red", "green", "purple", "orange", "blue", "black", "grey", "gold"] #list of colours for the graph lines 
):

    data_dict = {}
    precision_labels = [] 

    model_sizes = {
        'qat': [qat_model_sizes, qat_path], 
        'ptq': [ptq_model_sizes, ptq_path]
    }

    for training_type in ['qat', 'ptq']: 

        for nodes in model_sizes[training_type][0]: 

            #append data to a list in a dictionary. The associated key will be the name shown on the graph legend.
            data_dict[f"{nodes}-node {training_type.upper()} model"] = []
            selected_list = data_dict[f"{nodes}-node {training_type.upper()} model"]

            for i in range(precision_range[0],precision_range[1]): 

                if bit_to_set == 1: 
                    spec_path = os.path.join(model_sizes[training_type][1], f"{nodes}.{i}.{set_bit}.{name}/")
                elif bit_to_set == 0:
                    spec_path = os.path.join(model_sizes[training_type][1], f"{nodes}.{set_bit}.{i}.{name}/") 
                else: 
                    raise ValueError("Value of bit to set not recognized.")

                if os.path.exists(spec_path): 

                    mod_data = np.loadtxt(os.path.join(spec_path, "hls_predictions.txt")) 
                    mod_keras_pred = np.loadtxt(os.path.join(spec_path, "keras_predictions.txt"))

                    if graph_type == "mpe": 
                        #avoid divide by zero
                        nonzero_mask = mod_keras_pred != 0 
                        mod_result = np.abs(mod_data[nonzero_mask] - mod_keras_pred[nonzero_mask])/(np.abs(mod_keras_pred[nonzero_mask]))*100 
                        #If desired, you can print out the number of samples you've excluded. 
                        print("Number of samples excluded: ", np.sum(~nonzero_mask)) 

                    elif graph_type =="mae":
                        mod_result = np.abs(mod_data - mod_keras_pred)
                    
                    else: 
                        raise ValueError(f"Graph type {graph_type} not recognized.")
                    
                    mn_result = np.mean(mod_result) 
                    selected_list.append(mn_result)  

                    if (nodes == qat_model_sizes[0] or nodes == ptq_model_sizes[0]) and len(precision_labels) < (precision_range[1] - precision_range[0]):
                        #create x axis labels for graphing 
                        if bit_to_set == 1: 
                            precision_labels.append(f"ap_fixed<{i},{set_bit}>") 
                        else: 
                            precision_labels.append(f"ap_fixed<{set_bit},{i}>")

                else:
                    print(f"{training_type} model path {spec_path} not found.")         

    #set one line for each model 
    num_lines = len(data_dict) 
    data_names = []
    plt.figure(figsize=(12,7))

    #append the names of the models
    for keys in data_dict: 
        data_names.append(keys)

    for i in range(num_lines): 

        y = data_dict[data_names[i]]
        x = np.arange(len(y))

        if "QAT" in data_names[i]: 
            plt.plot(x, y, color=colours[i%len(colours)], linestyle="dashed", label=data_names[i])
            print(f"QAT line {i} successfully generated.")

        else: 
            plt.plot(x, y, color=colours[(i - len(qat_model_sizes))%len(colours)], label=data_names[i])
            print(f"PTQ line {i} successfully generated.")

    #Generate the plot. 
    plt.legend() 
    plt.title(plt_title)
    plt.text(x=ax.get_xlim()[0], y=ax.get_ylim()[1], s=f"Number of samples per precision: {num_lines}", ha='left', va='top')
    plt.xlabel(xlabel)
    plt.xticks(ticks=x.flatten(), labels=precision_labels, rotation=45, ha="right")
    plt.ylabel(graph_type.upper())
    plt.tight_layout()
    plt.show() 

    im_path = os.path.join(save_path + f"{plt_title}.png")
    plt.savefig(im_path)
    plt.close()

    print(f"{plt_title} plot generated and saved to {im_path}.")

#Reduces the range of a plt cmap for better viewing
def truncate_colormap(cmap_in, minval=0.3, maxval=0.7, n=256):
    new_cmap = LinearSegmentedColormap.from_list(
        f"trunc({cmap_in.name},{minval:.2f},{maxval:.2f})",
        cmap_in(np.linspace(minval, maxval, n))
    )
    return new_cmap


#helper function for sweep_heatmap 
def find_min_idx(
    array: np.array, 
    max_rows: int, 
    max_cols: int, 
    mask: int
): 
    
    flat_copy = array.flatten()
    min_val = flat_copy.flatten()[0] 
    if min_val == mask: 
        for i in range(len(flat_copy)): 
            if flat_copy[i] != mask: 
                min_val =flat_copy[i] 
                break 
    tgt_row = 0 
    tgt_col = 0 

    for i in range(max_rows): 
        for j in range(max_cols): 
            if array[i][j] < min_val and array[i][j] != mask: 
                min_val = array[i][j]
                tgt_row = i 
                tgt_col = j 

    return [min_val, tgt_row, tgt_col]


#Function: Plots a 2D precision sweep between integer and decimal bits - used to select the optimal precision combination 
def sweep_heatmap(
    model_nodes: int, 
    data_path: str, #path to general folder containing the tensor file folders
    plt_title: str, 
    int_range: list, 
    full_range: list, 
    graph_type: str, #either 'mpe' or 'mae'
    save_path: str, 
    name: str='model', #name of the model references in the sweep directory 
    max_percentile: int=100, #percentile to cut off at. 
    cmap_range: list=[0, 1], 
    num_samples: int=1000, 
    cmap: str='RdBu', 
    x_label: str= "Total Number of Bits", 
    y_label: str="Integer Bits", 
    square: bool=True, 
    make_txt: bool=True #offers the option to generate a text file with numerical results. 
): 

    from matplotlib.colors import LogNorm

    mask_val = -99.0 

    num_rows = int_range[1] - int_range[0]
    num_cols = full_range[1] - full_range[0]
    results = np.ones((num_rows, num_cols))*(mask_val)  # [rows = int_bits, cols = full_bits]. -99 used for masking. 

    for full_bits in range(full_range[0], full_range[1]): 

        for int_bits in range(int_range[0], int_range[1]): 

            if int_bits > full_bits: 
                continue

            model_name = f"{model_nodes}.{full_bits}.{int_bits}.{name}/"
            model_path = os.path.join(data_path, model_name)

            if not os.path.exists(model_path):
                print(f"Warning: Missing {model_path}")
                continue

            #Calculate mpe 
            if graph_type == 'mpe': 

                keras_preds = np.loadtxt(os.path.join(model_path, "keras_predictions.txt"))[:num_samples]
                hls_preds = np.loadtxt(os.path.join(model_path, "hls_predictions.txt"))[:num_samples]
                pe = np.abs(hls_preds - keras_preds)/np.abs(keras_preds) * 100
                mpe = np.mean(pe)
                results[int_bits - int_range[0]][full_bits - full_range[0]] = mpe 

            #Calculate mae
            elif graph_type == 'mae': 

                keras_preds = np.loadtxt(os.path.join(model_path, "keras_predictions.txt"))[:num_samples]
                hls_preds = np.loadtxt(os.path.join(model_path, "hls_predictions.txt"))[:num_samples]

                err = np.abs(hls_preds)/np.abs(keras_preds)
                m_err = np.mean(err)

                results[int_bits - int_range[0]][full_bits - full_range[0]] = m_err

            else: 
                raise ValueError("Graph type not recognized. Edit the function and try again.")

    if make_txt == True: 
        with open(os.path.join(data_path, f"results_{model_nodes}_{graph_type}.txt"), 'w') as out_file:

            min_val, min_row, min_col = find_min_idx(results, num_rows, num_cols, mask_val)

            out_file.write(f"Best performing precision: ap_fixed<{min_col + full_range[0]},{min_row + int_range[0]}>\n")
            out_file.write(f"Corresponding {graph_type}: {min_val}\n")
            out_file.write("-----------\n")

            for int_bits in range(num_rows): 
                for full_bits in range(num_cols): 
                    if results[int_bits][full_bits] != -99.0: 
                        if graph_type == 'mpe': 
                            out_file.write(f"MPE for ap_fixed<{full_bits + full_range[0]},{int_bits+int_range[0]}>: {results[int_bits][full_bits]}\n")
                        else: 
                            out_file.write(f"HLS/Keras pred for ap_fixed<{full_bits + full_range[0]},{int_bits+int_range[0]}>: {results[int_bits][full_bits]}\n")

    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot()
    cmap = truncate_colormap(plt.get_cmap(cmap), cmap_range[0], cmap_range[1])

    #make results fit the graph 
    temp = results.flatten() 
    results = np.flipud(results) 
    mask = results < 0
    yticks = np.arange(results.shape[0])

    ytick_labels = [] #integer bits vary along y 
    xtick_labels = [] #total number of bits vary along x 
    for int_b in np.flip(np.arange(int_range[0], int_range[1])): 
        ytick_labels.append(f"ap_fixed<{full_range[0]}, {int_b}>")
    for full_b in np.arange(full_range[0], full_range[1]):
        xtick_labels.append(f"ap_fixed<{full_b}, {int_range[0]}>")
        
    if graph_type == 'mpe': 

        cbar_kws = {'label': 'Mean Percent Error', 'pad': 0.03}
        vmax = np.percentile(temp, max_percentile) 
        hm = sns.heatmap(data=results, mask=mask, vmin=0, vmax=vmax, cmap=cmap, norm=LogNorm(), cbar=True, cbar_kws=cbar_kws, square=False)
        #hm = sns.heatmap(data=results, mask=mask, vmin=0, vmax=vmax, cmap=cmap, cbar=True, cbar_kws=cbar_kws, square=False) #non-log normalized. 
        plt.text(x=ax.get_xlim()[0], y=ax.get_ylim()[1], s=f"Max percentile of data: {max_percentile}%\nGraph type: {graph_type}", ha='left', va='top')

    else: 

        cbar_kws = {'label': 'HLS Pred/Keras Pred', 'pad': 0.075} 
        vmax = np.percentile(temp, max_percentile) 
        hm = sns.heatmap(data=results, mask=mask, vmin=0, vmax=vmax, cmap=cmap, norm=LogNorm(), center=1, cbar=True, cbar_kws=cbar_kws, square=False)
        plt.text(x=ax.get_xlim()[0], y=ax.get_ylim()[1], s=f"Graph type: {graph_type}\n", ha='left', va='top')

    hm.set_title(plt_title, fontsize=16)
    hm.set_xlabel(x_label, fontsize=16)
    hm.set_xticks(np.arange(num_cols))
    hm.set_xticklabels(xtick_labels, rotation=30)

    if square == True: 
        hm.set_aspect("equal")

    hm.set_ylabel(y_label, fontsize=16)
    hm.set_yticks(yticks) 
    hm.set_yticklabels(ytick_labels, rotation=0)

    plt.tight_layout()

    plt.savefig(os.path.join(save_path, f"{plt_title}.png"))
    print(f"{plt_title} heatmap saved under {save_path}.")


#Analyzes per-layer outputs of a keras model so you can determine if different layers can use differing precisions 
def trace_keras_model(
    model_path: str, #expects .keras file 
    custom_objects: dict, 
    test_data: np.array, #compatible with the load_test_data function as defined in utils.py  
    num_samples: int=10000
): 

    kmodel = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects) 
    layer_outputs = [layer.output for layer in kmodel.layers]
    tmp_model = tf.keras.models.Model(inputs=kmodel.input, outputs=layer_outputs)

    inputs = test_data[:num_samples]
    tmp_outputs = tmp_model.predict(inputs) #this is a list of arrays. 

    keras_trace = {}
    for i, layer in enumerate(kmodel.layers): 
        keras_trace[layer.name] = tmp_outputs[i]

    return keras_trace 

#Simple function to interpret the .pickle keras training history file.
def print_training_history(
    history_path: str, #path to pickle file
    keys_to_analyze: list
): 

    with open(history_path, 'rb') as file: 
        history = pickle.load(file) 

        print("History dictionary keys: ")
        for i, key in enumerate(history.keys()): 
            if i == 0:
                print("Number of training epochs: ", len(history[key]))
            print(key)

        for k in keys_to_analyze: 
            print(f"Training history for {k}:")
            
            for epoch in range(len(history[k])): 
                print(f"Epoch {epoch+1} {k}: ", history[k][epoch])


#Analyzes the per-layer outputs of model traces and outputs into a text file (txt_file)  
#This function assumes we are comparing the same model, just different ptq/qat versions   
def analyze_trace(
    trace_dicts: dict, #dictionary of dictionaries. 
    txt_file: str="trace_analysis.txt", 
    sample_num: int=0, #idx of sample. 
): 

    maxlength = 0
    for key in trace_dicts.keys():

        #store the longest key for use later. 
        if len(trace_dicts[key]) > maxlength: 
            maxlength = len(trace_dicts[key])
            longest_dict = key 

        print(f"{key} keys:")
    
        for k in trace_dicts[key].keys():
            print(k)

    ckeys = get_common_keys(trace_dicts, longest_dict)
    first_dict = next(iter(trace_dicts))
  
    with open(txt_file, "w") as out_file: 
        col_widths = []
        for common_layer in ckeys: 
            out_file.write(f"Common layer: {common_layer}\n") 
            print("Common layer: ", common_layer)

            #write the headings (the dictionary names)
            for trace in trace_dicts.keys(): 
                out_file.write(f"{trace} outputs:           | ")
                col_widths.append(len(f"{trace} outputs:           |") - 2) #not sure why but this works. 
            out_file.write("\n")

            #write the data row-wise, where each row corresponds to the same node (and same sample) for the different dictionaries. Only do 1 sample or it gets overwhelming. 
            #sample_num tells you the number of the sample you are inspecting.     
            #This function assumes we are comparing the same model, just different ptq/qat versions, thus the number of nodes is the same. 
            node_arrays = []

            for _dict in trace_dicts.keys(): #append the node outputs for this specific layer for all trace dictionaries.           
                node_arrays.append(trace_dicts[_dict][common_layer][sample_num].flatten())

            for node in range(len(node_arrays[0])): 
                for _dict in range(len(node_arrays)): 
                    val = node_arrays[_dict][node]
                    out_file.write(f"{val:>{col_widths[_dict]}.8f} | ")
                out_file.write("\n")
            out_file.write("\n----------\n")

    print("Analysis txt file save path: ", txt_file)


#Returns: the list of keys with common names between dictionaries. (Helper function for analyze_trace.)
def get_common_keys(
    trace_dicts: dict, 
    longest_dict_name: str="one_dict", #refers to the longest dictionary in the dictionary of dicts
): 

    if longest_dict_name == "one_dict": 
        dict_name = next(iter(trace_dicts))
        longest_dict = trace_dicts[dict_name]
    else: 
        longest_dict = trace_dicts[longest_dict_name]

    common_keys = []

    for key_name in longest_dict: 
        is_common = 1
        for dict_ in trace_dicts.keys(): #loop through all the dictionaries to see if the common key is present. 
            if key_name not in trace_dicts[dict_]: 
                is_common = 0
        if is_common == 1: 
            common_keys.append(key_name)
    
    return common_keys

#Returns a scatter plot to show the signed difference between layers for the Keras and hls models; used to pinpoint problematic layers. 
def per_layer_analysis(
    hls_dict_path: str, 
    keras_dict_path: str, 
    plt_title: str, 
    save_path: str, 
    samples: int = 100, 
    colour: str='royalblue', 
): 

    with open(keras_dict_path, 'rb') as file: 
        keras_dict = pickle.load(file) 

    with open(hls_dict_path, 'rb') as file: 
        hls_dict = pickle.load(file) 

    layer_names = []

    fig, ax = plt.subplots(figsize=(15, 8))

    for i, key in enumerate(hls_dict.keys()): 

        print(f"Working on layer {key}...") 
        layer_names.append(key)   

        for sample in range(samples): 

            keras_x = keras_dict[key][sample]
            hls_x = hls_dict[key][sample]

            difference = (hls_x - keras_x).flatten()
            ax.scatter([i] * len(difference), difference, color=colour, alpha=0.5, s=50)
        
        print(f"Layer {key} scatter generated.") 

    ax.hlines(y=0, xmin=0, xmax=21, color="black", linestyle="--")
    ax.set_xticks(np.arange(len(layer_names)))
    ax.set_xticklabels(layer_names, rotation=45, ha="right")
    ax.set_xlabel("Layer name")
    ax.set_ylabel("Difference per layer")
    ax.set_title(plt_title)

    plt.tight_layout()
    fig.tight_layout()
    plt.savefig(save_path + f"{plt_title}")
    plt.close()

    print(f"{plt_title} scatter generated and saved to {save_path}.")


