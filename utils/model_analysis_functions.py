"""This script contains functions I developed to analyze the difference between a Keras model 
and various hls4ml parameterizations. Performance metrics for analysis were mostly MPE and MAE, although functions can be 
easily modified to accomodate other methods, such as predictions / truth. Note that certain functions rely on naming 
conventions I developed as well as setting trace to True during hls4ml conversion - see the PTQ and QAT folders for more details."""

import numpy as np 
import pickle
import os 
import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import tensorflow as tf 


#used to visualize a 1D precision sweep between QAT and PTQ models, plotting MPE from the equivalent Keras model. 
#requires the original Keras model to be trained first. 
def qat_vs_ptq(
    qat_path: str, #path to folder containing results files 
    ptq_path: str,
    qat_model_sizes: list,
    ptq_model_sizes: list, 
    save_path: str, 
    precision_range: list, #upper bound is exclusive; list represents the range of the precision sweep for the changing bit 
    bit_to_set: int, #for ap_fixed<M,N>, fixing M -> bit_to_set = 0 and fixing N -> bit_to_set = 1
    set_bit: int, #the value of the bit that is set 
    name: str, #name for this particular sweep directory 
    num_samples: int=1000, 
    plt_title: str="PTQ vs. QAT MPE from equivalent keras model", 
    xlabel: str="Precision", 
    graph_type="mpe", #or mae. 
    colours = ["red", "green", "purple", "orange", "blue", "black", "grey", "gold"] #default list of colours for plotting. 
):

    data_dict = {}
    precision_labels = [] 

    model_sizes = {
        'qat': [qat_model_sizes, qat_path], 
        'ptq': [ptq_model_sizes, ptq_path]
    }

    for training_type in ['qat', 'ptq']: 

        for nodes in model_sizes[training_type][0]:

            data_dict[f"{nodes}-node {training_type.upper()} model"] = []
            selected_list = data_dict[f"{nodes}-node {training_type.upper()} model"]

            for i in range(precision_range[0],precision_range[1]): 

                if bit_to_set == 1: 
                    spec_path = os.path.join(model_sizes[training_type][1], f"{nodes}.{i}.{set_bit}.{name}/")
                elif bit_to_set == 0:
                    spec_path = os.path.join(model_sizes[training_type][1], f"{nodes}.{set_bit}.{i}.{name}/") 
                else: 
                    raise ValueError("Bit to set is an invalid value. Accepted values are 0 or 1.")

                if os.path.exists(spec_path): 

                    mod_data = np.loadtxt(os.path.join(spec_path, "hls_predictions.txt")) 
                    mod_keras_pred = np.loadtxt(os.path.join(spec_path, "keras_predictions.txt"))

                    if graph_type == "mpe": 

                        nonzero_mask = mod_keras_pred != 0 
                        mod_result = np.abs(mod_data[nonzero_mask] - mod_keras_pred[nonzero_mask])/(np.abs(mod_keras_pred[nonzero_mask]))*100 
                        #print("Number of samples excluded: ", np.sum(~nonzero_mask)) #optional for debugging. 

                    elif graph_type =="mae":

                        mod_result = np.abs(mod_data - mod_keras_pred)
                    
                    else: 
                        raise ValueError("Graph type not recognized. Edit function and try again.")
                    
                    mod_result = np.mean(mod_result) 
                    selected_list.append(mod_result)  

                    if (nodes == qat_model_sizes[0] or nodes == ptq_model_sizes[0]) and len(precision_labels) < (precision_range[1] - precision_range[0]):
                        #create x axis labels for graphing 
                        if bit_to_set == 1: 
                            precision_labels.append(f"<{i},{set_bit}>") 
                        else: 
                            precision_labels.append(f"<{set_bit},{i}>")

                else:
                    print(f"{graph_type} path {spec_path} not found.")

    num_lines = len(data_dict) 
    print("Data dictionary: ") 

    last_key = list(data_dict.keys())[-1]

    data_names = []
    fig, ax = plt.subplots(figsize=(12,7))

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


    ax.legend() 
    ax.set_title(plt_title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(graph_type.upper())
    ax.set_xticks(x)
    ax.set_xticklabels(precision_labels, rotation=45, ha="right")
    ax.text(x=ax.get_xlim()[0], y=ax.get_ylim()[1], 
            s=f"Number of samples used for predictions: {num_samples}", 
            ha='left', va='top')

    plt.tight_layout()

    im_path = os.path.join(save_path + f"{plt_title}.png")
    plt.savefig(im_path)
    plt.close()

    print(f"{plt_title} plot generated and saved to {im_path}.")

#Helper function for sweep_heatmap function 
def truncate_colormap(cmap_in, minval=0.3, maxval=0.7, n=256):
    new_cmap = LinearSegmentedColormap.from_list(
        f"trunc({cmap_in.name},{minval:.2f},{maxval:.2f})",
        cmap_in(np.linspace(minval, maxval, n))
    )
    return new_cmap


#Helper function for sweep_heatmap function 
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

#plots the results of a 2D hls4ml precision sweep for a single Keras model
#total precision increases along x and integer precision increases along y 
def sweep_heatmap(
    model_nodes: int, 
    data_path: str, #path to general folder containing the tensor file folders
    plt_title: str, 
    int_range: list, 
    full_range: list, 
    graph_type: str, #either 'mpe' or 'mae'
    name: str, #name of the model for the sweep.
    max_percentile: int=100, #percentile to cut off at. 
    cmap_range: list=[0, 1], 
    num_samples: int=1000, 
    cmap: str='RdBu', 
    save_path: str="/home/ali2/hls4ds/training_programs/wk4/regression_only/", 
    x_label: str= "Total Number of Bits", 
    y_label: str="Integer Bits", 
    square: bool=True, 
    gen_txt: bool=True #offers the option to generate a text file with numerical results for analysis. 
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

            if graph_type == 'mpe': 

                keras_preds = np.loadtxt(os.path.join(model_path, "keras_predictions.txt"))[:num_samples]
                hls_preds = np.loadtxt(os.path.join(model_path, "hls_predictions.txt"))[:num_samples]

                pe = np.abs(hls_preds - keras_preds)/np.abs(keras_preds) * 100
                mpe = np.mean(pe)

                print(f"MPE for {model_name}: {mpe}")

                results[int_bits - int_range[0]][full_bits - full_range[0]] = mpe 

            elif graph_type == 'mae': 

                keras_preds = np.loadtxt(os.path.join(model_path, "keras_predictions.txt"))[:num_samples]
                hls_preds = np.loadtxt(os.path.join(model_path, "hls_predictions.txt"))[:num_samples]

                err = np.abs(hls_preds - keras_preds)
                m_err = np.mean(err)

                results[int_bits - int_range[0]][full_bits - full_range[0]] = m_err

            else: 
                raise ValueError("Graph type not recognized. Edit the function and try again.")

    if gen_txt: #generate text file. 

        with open(os.path.join(data_path, f"results_{model_nodes}_{graph_type}.txt"), 'w') as out_file: #use w so we don't just make a giant file. 

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

    #fit results to the graph. 
    temp = results.flatten()
    results = np.flipud(results) #flip vertically 
    mask = results < 0
    yticks = np.arange(results.shape[0])

    ytick_labels = []
    for int_b in np.flip(np.arange(int_range[0], int_range[1])): 
        ytick_labels.append(f"<{full_range[0]}, {int_b}>")
    
    xtick_labels = []
    for full_b in np.arange(full_range[0], full_range[1]): 
        xtick_labels.append(f"<{full_b}, {int_range[0]}>")

    if graph_type == 'mpe': 

        cbar_kws = {'label': 'Mean Percent Error', 'pad': 0.03}
        vmax = np.percentile(temp, max_percentile) 
        hm = sns.heatmap(data=results, mask=mask, vmin=0, vmax=vmax, cmap=cmap, norm=LogNorm(), cbar=True, cbar_kws=cbar_kws, square=False)
        #hm = sns.heatmap(data=results, mask=mask, vmin=0, vmax=vmax, cmap=cmap, cbar=True, cbar_kws=cbar_kws, square=False) #non-logNorm version 
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
    

#Analyzes the per-layer output of a Keras model; useful for deciding if some layers require more precision than others. 
def trace_keras_model(
    model_path: str, #expects .keras file 
    custom_objects: dict, 
    test_data: np.array, #compatible with the load_test_data function as defined in utils.py  
    num_samples: int=100
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

#Used to analyze, side-by-side, the trace dictionaries of 2+ models (typically Keras and its hls4ml equivalent)
#This function assumes we are comparing the same model (aka same number of nodes), just different versions
def analyze_trace(
    trace_dicts: dict, #dictionary of dictionaries. 
    sample_num: int=0, #idx of sample. 
    txt_file: str="trace_analysis.txt" #title of text file to be generated. 
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
    print("Ckeys: ", ckeys)
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


#Returns: the list of keys with common names between dictionaries. 
#Helper function for analyze_trace. 
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

#Returns a spread plot showing the difference between the Keras and hls4ml models - useful to determine if there are problematic 
#layers, or if error is cumulative. 
def per_layer_analysis(
    hls_dict_path: str, 
    keras_dict_path: str, 
    save_path: str, 
    plt_title: str, 
    samples: int=1000, 
    colour: str='royalblue'
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

    print(f"{plt_title} plot generated and saved to {save_path}.")

