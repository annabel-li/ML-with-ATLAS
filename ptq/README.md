# Introduction to the PTQ folder  

## Quick facts 

'Post-Training Quantization' refers to the process of converting floats to fixed-point representations after a model has been trained in software
with float32 representations. Working with PTQ, I studied the impacts that model size (# of nodes per dense layer) and post-training quantization 
(number of bits allocated to the integer and decimal portions of numbers in the model) affected model accuracy. 

## Workflow 

Step 1 - train the model in Python with Tensorflow and Keras 
