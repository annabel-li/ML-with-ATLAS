# Introduction to the PTQ folder  

## Quick facts 

'Post-Training Quantization' refers to the process of converting floats to fixed-point representations after a model has been trained in software
with float32 representations. Working with PTQ, I studied the impacts that model size (# of nodes per dense layer) and post-training quantization 
(number of bits allocated to the integer and decimal portions of numbers in the model) affected model accuracy. 

## Workflow 

#### Step 1: train the model in Python with Tensorflow and Keras using ```trainModel.py``` 

<i>Note: as I am not the owner of the data processing function I used, I've generated dummy data for the sake of the script. 
We trained our model on ATLAS/Root data with a set "vmax" (maximum number of cells, or input vector length,) as Keras expects all inputs to be 
the same shape. Input clusters with less data than vmax were padded up to the correct amount with -6, a number decided on based on 
the accuracy that could be maintained in the hls4ml-version of the KSum layer. Otherwise, entries in the input vectors were log-normalized and divided by 10.</i>

#### Step 2: Convert the model with hls4ml, using ```convertModel.py``` 

Set up the hls model config with the desired test precision. The script offers the option to convert models with multiple precision combinations, varying integer and full bit width, as well as setting different precisions for intermediate + output vectors and weights + biases. 

<i>Note: To make our custom KSum layer hls4ml compatible, we had to set up Extension API. While I was also not in charge of developing this code and therefore cannot share it here, there is excellent documentation about how to do so on the official [hls4ml website](https://fastmachinelearning.org/hls4ml/advanced/extension.html).</i> 


#### Step 3: Test the Keras and hls4ml versions of the model with the same input samples, and compare outputs 

As the hls4ml model predicted significantly slower, the maximum number of input samples passed was 1000. The ```convertModel.py``` script automatically sets up per-layer output tracing and saves the model prediction results in a text file under the model's name. Results were analyzed with functions from the ```model_analysis_funcs.py``` file under ```ML-with-ATLAS/utils/```.  



