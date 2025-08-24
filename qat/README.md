# Introduction to the QAT folder 

## Quick facts 

Quantization-Aware Training is the process of "exposing" models to lower-precision operations when training, allowing them to learn parameters that are 
more robust to hardware quantizations. Similar to PTQ, I tested model size and precision combination sweeps, and compared the results to the original 
Keras model as well as the PTQ equivalent. I found that while QAT outperforms PTQ at lower bit widths (from 18 - 22 total bit width for vector and 
weights and biases representations), it is far more inconsistent, showing spikes where PTQ performance stabilizes at higher precisions. This was analyzed 
using the ```qat_vs_ptq()``` function in ```utils/model_analysis_funcs.py```. 

## Required libraries 

Training with QAT requires QKeras, which can be installed with ```pip install qkeras```. It may be best to create a virtual python environment when 
doing so as I ran into compatibility issues with the versions of NumPy and tensorflow I already had installed, and had to uninstall keras and 
reinstall ```tf_keras``` (legacy keras) specifically. This also required altering some of the files in the hls4ml package when it came to searching for and importing keras. 

