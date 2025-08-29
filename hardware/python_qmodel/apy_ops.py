import numpy as np 
import apytypes as apy 
from apytypes import APyFixedAccumulatorContext

def expand2D(mat): 
    if mat.ndim < 2: #don't use expand dims or this would convert it to an np array. 
        return mat.reshape((1, -1))
    else: 
        return mat 

def apyMatMul2D(mat1, mat2, bits, frac_bits): #2 apy mats 

    mat1 = expand2D(mat1) 
    mat2 = expand2D(mat2)

    with APyFixedAccumulatorContext(bits=bits, frac_bits=frac_bits, quantization=apy.QuantizationMode.RND): 
        res = mat1 @ mat2 

    return res 
