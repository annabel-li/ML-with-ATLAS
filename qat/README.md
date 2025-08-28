# Introduction to the QAT folder 

## Folder structure 

```
|-- qat/
    - README.md
    - convertModelQatKsum.py 
    - mass_convert.py
    - trainModelQat.py
    - upper_level_batch.sh
```

## Quick facts 

Quantization-Aware Training is the process of exposing models to lower-precision operations when training, allowing them to learn parameters that are more robust to hardware quantizations. Similar to PTQ, I tested model size and precision combination sweeps, and compared the results to the original Keras model as well as the PTQ equivalent. I found that while QAT outperforms PTQ at lower bit widths (from 18 - 22 total bit width for vector and weights and biases representations), it is far more inconsistent, showing spikes where PTQ performance stabilizes at higher precisions. 

This was analyzed using the ```qat_vs_ptq()``` function in ```utils/model_analysis_funcs.py```. 

## Required libraries 

Training with QAT requires QKeras, which can be installed with ```pip install qkeras```. It may be best to create a virtual python environment when 
doing so as I ran into compatibility issues with the versions of NumPy and Tensorflow I already had installed, and had to uninstall keras and 
reinstall ```tf_keras``` (legacy keras) specifically. 

This also required altering some of the files in the hls4ml package when it came to searching for and importing keras. If you get an error message from hls4ml saying ```"NameError: name 'keras' is not defined. Did you mean: 'qkeras'?``` you may fix it by following the hls4ml path in the error message to ```profiling.py``` and changing ```import keras``` to ```import tf_keras as keras```. 

To avoid serialization issues, I also converted QKeras models to their hls4ml forms in the same workflow instead of saving and then re-loading.

## Workflow & how to use the scripts

I developed these scripts to mass-convert a series of Keras models to see the effects of different parameterizations, namely precision, on the equivalent QAT hls4ml performance. To run, change the arguments for precision in ```upper_level_batch.sh```, which will call the ```mass_convert.py``` script and the required training and converion functions in ```trainModelQat.py``` and ```convertModelQatKsum.py```. 

The workflow is automated, so the only file that needs to be interacted with is ```upper_level_batch.sh```. 


