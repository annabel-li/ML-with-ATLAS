# Introduction to the QAT folder 

## Quick facts 

Quantization-Aware Training is the process of "exposing" models to lower-precision operations when training, allowing them to learn parameters that are 
more robust to hardware quantizations. Similar to PTQ, I tested model size and precision combination sweeps, and compared the results to the original 
Keras model as well as the PTQ equivalent. I found that while QAT outperforms PTQ at lower bit widths (from 18 - 22 total bit width for vector and 
weights and biases representations), it is far more inconsistent, showing spikes where PTQ performance stabilizes at higher precisions. This was analyzed 
using the ```qat_vs_ptq()``` function in ```utils/model_analysis_funcs.py```. 

## Required 
