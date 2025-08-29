from nn_utils import Dense_Layer, KSum_Layer, Regression_Layer
import numpy as np 
import tensorflow as tf
import sys 
import os 
from time import time

sys.path.append(os.path.join(os.path.dirname(__file__), '/home/ali2/hls4ds/utils/deepsetModels/custom_layers'))
from init_layers import KSum 


# Work in progress: 
# 1 - if you build the model and then change the general precision for vectors or weights/biases, the model is not rebuilt to reflect those changes.  

def is_dense_activation(layer_config): 

    is_activation = False
    activations = ['relu', 'tanh', 'sigmoid']

    for key in layer_config.keys(): 
        if 'activation' in key: 
            if layer_config[key] in activations: 
                is_activation = True 
    
    return is_activation 

class DS_Model: 

    def __init__(
        self, 
        keras_file, #saved .keras model from training 
        vec_precision={'n_word': 16, 'n_frac': 10}, #dictionaries. 
        wb_precision={'n_word': 16, 'n_frac': 10}
    ): 

        self.keras_file = keras_file 
        self.vec_precision = vec_precision
        self.wb_precision = wb_precision
        self.vfull_bits = vec_precision['n_word']
        self.vfrac_bits = vec_precision['n_frac']
        self.wbfull_bits = wb_precision['n_word']
        self.wbfrac_bits = wb_precision['n_frac']


        self.build_model() #build the model

    def load_model(self): 

        return tf.keras.models.load_model(self.keras_file, compile=False, custom_objects={"KSum": KSum})

    def get_params_from_keras(self): 

        self.loaded_model = self.load_model()
        model_params = [] #list of dictionaries, where each layer's name, weights (if applicable), and shape is stored in the dict. 
        model_config = self.loaded_model.get_config()

        for i, layer in enumerate(self.loaded_model.layers):         
            layer_config = layer.get_config()

            if i == 0: #skip input layer 
                continue

            if len(layer.get_weights()) != 0: #if the length of the layer weights list is not zero... 
                layerweights = layer.get_weights()[0]
                layerbias = layer.get_weights()[1]

            else: #layer has no weights 
                layerweights = 'None' 
                layerbias = 'None' 

            if layer.name.lower() == "ksum": 
                classname = "custom"
            else: 
                classname = model_config['layers'][i]['class_name']

            model_params.append({
                'layername': layer.name, 
                'inputshape': list(layer.input_shape),
                'layerweights': layerweights, 
                'layerbias': layerbias, 
                #'is_activation': is_dense_activation(layer_config)
                'classname': classname 
            })
        print("model params acquired.")
        self.loaded_params = model_params 


    def build_model(self): #returns a list with layer objects 

        self.get_params_from_keras() #get parameters from trained keras model
        model = {} #dict with layer objects - using dict so it's easy for users to adjust the precision of layers 

        for i in range(len(self.loaded_params)): 
            layer = self.loaded_params[i] #current layer 
            layernum = i 

            if layer['classname'].lower() == "activation": #skip activation layers because we calculate Relu as a part of the dense layer. 
                continue 

            elif layer['layername'] == 'KSum': #KSum layer 
                newLayer = KSum_Layer(layernum=layernum, vfull_bits = self.vfull_bits, vfrac_bits=self.vfrac_bits) 

            else:  
                nodes = np.shape(layer['layerweights'])[-1] #last dimension of the weights matrix. 
                weights = layer['layerweights']
                bias = layer['layerbias']
                layername=layer['layername']
                if i == len(self.loaded_params) - 1: #regression layer 
                    newLayer = Regression_Layer(weights=weights, bias=bias, layernum=layernum, vfull_bits=self.vfull_bits, vfrac_bits=self.vfrac_bits, wbfull_bits=self.wbfull_bits, wbfrac_bits=self.wbfrac_bits)
                
                else: #Dense layer
                    newLayer = Dense_Layer(nodes=nodes, 
                                            weights=weights, 
                                            bias=bias, 
                                            layername=layername, 
                                            layernum=layernum, 
                                            vfull_bits=self.vfull_bits, 
                                            vfrac_bits=self.vfrac_bits, 
                                            wbfull_bits=self.wbfull_bits, 
                                            wbfrac_bits=self.wbfrac_bits)

            model[f'{newLayer.layername}'] = newLayer

        self.model = model
        print("Model built.")


    def print_attr(self):
        print("-" * 80)
        header = f"{'Layer':<15}{'Nodes':<15}{'Weights Shape':<20}{'Weights/Bias Prec':<25}{'Output Prec':<20}"
        print(header)
        print("-" * 80)

        for name, layer in self.model.items(): 
            layername = f"{name:<15}"
            nodes = f"{layer.nodes if layer.weights is not None else '-':<15}"
            weights_shape = f"{str(list(np.shape(layer.weights))) if layer.weights is not None else '-':<20}"
            if layer.weights is not None and layer.bias is not None:
                wb_prec = f"ap_fixed<{layer.wbfull_bits},{layer.wbfull_bits - layer.wbfrac_bits}>"
            else:
                wb_prec = "-"
            wb_prec = f"{wb_prec:<25}"

            out_prec = f"ap_fixed<{layer.vfull_bits},{layer.vfull_bits - layer.vfrac_bits}>"
            out_prec = f"{out_prec:<20}"

            print(f"{layername}{nodes}{weights_shape}{wb_prec}{out_prec}\n")



    def get_num_samples(self, in_data):
        if in_data.ndim == 2: 
            in_data = np.expand_dims(in_data, axis=2) #expand to 3D for consistency. 
            return 1
        else: 
            return np.shape(in_data)[0]

    def predict(self, in_data): 

        #self.build_model() #I think you need to do this in order to quantize the inputs because you didn't pass in the inputs when building the model. 
        # Nevermind... you don't.  
        print("Starting prediction for quantized model.")
        print("[", end='') #printing progress bar.
        nsamples = self.get_num_samples(in_data)
        model_results = []
        step = self.getstep(nsamples)
        
        start_time = time()
        for sample in range(nsamples):  

            if sample % step == 0: #update progress bar. 
                print("=", end='', flush="true")
            
            vecs_copy = in_data[sample]        
            for name, layer in self.model.items(): 
                #print("-", end="")
                layer.in_data = vecs_copy
                output = layer.forward_pass()
                vecs_copy = output 
            model_results.append(vecs_copy)

        end_time = time()
        self.predict_duration = (end_time - start_time)
        print(f"]] - {self.predict_duration/nsamples:.6f}s/sample")

        self.model_outputs = np.array(model_results)
        return self.model_outputs
    
    def getstep(self, nsamples, lower_limit: int=30): 
        if nsamples < lower_limit: 
            return -99 
        else: 
            return int(nsamples/lower_limit) #nsamples between print outs.  
    

    






        
