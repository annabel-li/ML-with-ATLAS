
import numpy as np 
import apytypes as apy 
from apytypes import APyFixedAccumulatorContext
from apy_ops import apyMatMul2D


# Work in progress: 
# 1 - if you build the model and then change the general precision for vectors or weights/biases, the model is not rebuilt to reflect those changes. 

class Dense_Layer(): 

    def __init__(
        self, 
        nodes, #int
        weights, #np.array
        bias, #np.array 
        layername, #str
        layernum, #int 
        vfull_bits = 16, #it's annoying, but i'm including this many arguments bc if you pass a dict only and try to modify it later it gets confusing. 
        vfrac_bits = 10, 
        wbfull_bits = 16, 
        wbfrac_bits = 10,  
        in_data=None  #array 
    ): 

        self.weights = weights #weights, as an np array of floats 
        if bias is not None: #bias, as an np array of floats 
            self.bias = bias.reshape((1, -1)) #in case it's N x 1 or 1 x N; just keep it consistent. 
        self.nodes = nodes 
        self.layername = layername
        self.layernum = layernum 
        self.activation = 'relu'
        self._in_data_floats = in_data #save the raw float value to perform quantizations on if precision changes. 

        #precision settings
        self.vfull_bits = vfull_bits 
        self.vfrac_bits = vfrac_bits 

        if self.weights is not None and self.bias is not None: 
            self.wbfull_bits = wbfull_bits
            self.wbfrac_bits = wbfrac_bits
            #initial quantization
            self.process_wb() # <------ IMPORTANT: To view quantized weights and biases, call self.wb
            
        #self.quantize_weights()
        
        if in_data is not None: 
            self.quantize_inputs(in_data) 
        
        
    #def quantize_weights(self): 

        #self.weights = apy.APyFixedArray.from_float(self.weights, bits=self.wbfull_bits, frac_bits=self.wbfrac_bits)
        #self.bias = apy.APyFixedArray.from_float(self.bias, bits=self.wbfull_bits, frac_bits=self.wbfrac_bits)
        #print("Weights and biases have been quantized.")


    #we need to ask the layer to re-quantize weights or biases when we change the precision, so we need to make them properties...
    # 
    # Vectors:  
    @property 
    def vfull_bits(self): 
        return self._vfull_bits 
    @vfull_bits.setter 
    def vfull_bits(self, value):  
        self._vfull_bits = value 
        if self._in_data_floats is not None and hasattr(self, "vfrac_bits") == True: 
            if self.vfrac_bits > self.vfull_bits: 
                raise ValueError("Number of fractional bits cannot exceed total number of bits for intermediate and output vectors.")
            self.quantize_inputs(self._in_data_floats) 

    @property 
    def vfrac_bits(self): 
        return self._vfrac_bits 
    @vfrac_bits.setter 
    def vfrac_bits(self, value): 
        self._vfrac_bits = value 
        if self._in_data_floats is not None and hasattr(self, "vfull_bits") == True: 
            if self.vfrac_bits > self.vfull_bits: 
                raise ValueError("Number of fractional bits cannot exceed total number of bits for intermediate and output vectors.")
            self.quantize_inputs(self._in_data_floats) 

    #WB: 
    @property 
    def wbfull_bits(self): 
        return self._wbfull_bits 
    @wbfull_bits.setter 
    def wbfull_bits(self, value): 
        self._wbfull_bits = value 
        if hasattr(self, "wbfrac_bits"): #check if wbfrac_bits has been initiatlized yet because if not then you can't process wb. 
            if self.wbfrac_bits > self.wbfull_bits: 
                raise ValueError("Number of fractional bits cannot exceed total number of bits for weights and biases.")
            self.process_wb() 

    @property 
    def wbfrac_bits(self): 
        return self._wbfrac_bits 
    @wbfrac_bits.setter 
    def wbfrac_bits(self, value): 
        self._wbfrac_bits = value 
        if hasattr(self, "wbfull_bits"): 
            if self.wbfrac_bits > self.wbfull_bits: 
                raise ValueError("Number of fractional bits cannot exceed total number of bits for weights and biases.")
            self.process_wb() 



    @property
    def in_data(self): 
        #print("Getting _.in_data...")
        return self._in_data #defines a private attribute -> "read only; you set it once and can't change it again."

    @in_data.setter #makes in_data both write and read 
    def in_data(self, value):
        self._in_data_floats = value 
        if value is not None: 
            self.quantize_inputs(value)
            if len(np.shape(self._in_data)) > 1: 
                self.in_data_rows = np.shape(self._in_data)[-2]
                self.in_data_cols = np.shape(self._in_data)[-1]
            else: 
                self.in_data_rows = np.shape(self._in_data)[0]
  
    def quantize_inputs(self, value): 

        axis = value.ndim - 1
        if value.ndim < 2: 
            ones = np.ones(1)
        else: 
            ones = np.ones((value.shape[0], 1))
        value = np.append(value, ones, axis=axis)
        self._in_data = apy.APyFixedArray.from_float(value, bits=self.vfull_bits, frac_bits=self.vfrac_bits)

        print(f"Inputs have been quantized.")

    def forward_pass(self): #calculates the matmul accums for all the nodes 

        """Old casting method:"""
        #matmul_result = apyMatMul2D(self._in_data, self.weights, bits=self.vfull_bits, frac_bits=self.vfrac_bits)
        #self.output = np.maximum(((matmul_result+self.bias).cast(bits=self.vfull_bits, frac_bits=self.vfrac_bits)), 0)  

        """concatenated wb method:"""   
        matmul_result = apyMatMul2D(self._in_data, self.wb, bits=self.vfull_bits, frac_bits=self.vfrac_bits)
        self.output = np.maximum(matmul_result, 0)
        #print(f"Forward pass completed successfully for layer {self.layername}.")
        return self.output #returns a numpy array but the values themselves are the correct float form (no extraneous decimals.) this is fine because it will be reverted 
                            #back to APy form as soon as it passes through another layer. 
    
    def process_wb(self): 
        if self.weights is None or self.bias is None: 
            self.wb = None 
        else: 
            wb = np.append(self.weights, self.bias, axis=0)
            self.wb = apy.APyFixedArray.from_float(wb, bits=self.wbfull_bits, frac_bits=self.wbfrac_bits)
            print(f"Weights & biases have been concatenated. Shape: {self.wb.shape}, type: {type(self.wb)}")



class Regression_Layer(Dense_Layer): 

    def __init__(self, weights, bias, layernum, in_data=None, wbfull_bits = 16, wbfrac_bits = 10, vfull_bits=16, vfrac_bits=10, layername="Regression"): 

        super().__init__(
            layername=layername, 
            nodes=1, 
            weights=weights,
            bias=bias, 
            layernum=layernum, 
            in_data=in_data, 
            vfull_bits=vfull_bits,
            vfrac_bits=vfrac_bits, 
            wbfull_bits = wbfull_bits, 
            wbfrac_bits = wbfrac_bits
        ) #call the parent class's constructors IN THE RIGHT ORDER. 

    def forward_pass(self): 

        matmul_result = apyMatMul2D(self._in_data, self.wb, bits=self.vfull_bits, frac_bits=self.vfrac_bits)
        self.output = apy.APyFixedArray.from_float(matmul_result, bits=self.vfull_bits, frac_bits=self.vfrac_bits) #wraps the output as an APy array. 
        return self.output



class KSum_Layer(Dense_Layer):

    def __init__(self, layernum, in_data=None, vfull_bits=16, vfrac_bits=10, layername="KSum"): 
        super().__init__(
            layername=layername, 
            layernum=layernum,
            nodes=None, 
            weights=None,
            bias=None, 
            in_data=in_data, 
            vfull_bits=vfull_bits,
            vfrac_bits=vfrac_bits, 
            wbfull_bits = None, 
            wbfrac_bits = None
        ) #call the parent class's constructors IN THE RIGHT ORDER. 

        self.default_precision = {
            'n_word': 16, 
            'n_frac': 10
        } #ap_fixed<16,6> 

    def forward_pass(self): 

        with APyFixedAccumulatorContext(bits=self.vfull_bits, frac_bits=self.vfrac_bits, quantization=apy.QuantizationMode.RND): #not sure why but this doesn't work for addition.
            self.output = self._in_data.sum(axis=0).cast(bits=self.vfull_bits, frac_bits=self.vfrac_bits) 
            #sum up all the vectors, which are stored as rows. 
            #print("Forward pass successfully completed for KSum.")
        
        return self.output #returns casted to the correct precision. 
    
    def quantize_inputs(self, value): #necessary to redefine or else there's just an extra 1 appended at the end of the vector... which we don't want! 
        self._in_data = apy.APyFixedArray.from_float(value, bits=self.vfull_bits, frac_bits=self.vfrac_bits)

    #non-applicable methods from the parent class: 
    #def relu_func(self): 
    #    raise NotImplementedError("Method not supported.")
    
    #def quantize_weights(self): 
    #    pass 

    def process_wb(self): 
        pass







        











