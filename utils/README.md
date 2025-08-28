## Folder Overview 

### Structure 

```
├── utils/ 
        └── base_models/
            - __init__.py
            - custom_layers.py 
            - fastjet_ksum.py 
            - fastjet_ksum_hgq.py 
            - fastjet_ksum_qat.py 
        - README.md
        - dummy_data_generator.py
        - model_analysis_funcs.py 
```

## Folder Content 

1) ```base_models/```
Contains base DeepSets model definition files for different kinds of training (basic Keras, QAT, and [HGQ](https://fastmachinelearning.org/hls4ml/advanced/hgq.html)), as well as the KSum layer class (found in ```custom_layers.py```). Model definitions are based on the 
work done at https://github.com/fastmachinelearning/l1-jet-id/blob/main/fast_jetclass/deepsets/deepsets_synth.py.

2) Scripts
- ```dummy_data_generator.py```: dummy data generating function as an example for this repo. The actual data generator function is not my property.
Our model was trained on log-normalized ROOT data that had 4 input features: cell eta, cell phi, calorimeter layer (of the hit) and the raw energy at that location, with
128 as the maximum number (vmax) of input cells - thus our models had an input size of (None, 128, 4). Clusters with ncells < vmax were padded up to vmax with -6.
- ```model_analysis_funcs.py```: script containing various functions I developed to analyze the performance of parameterizations of our hardware-friendly DeepSets model. Performance between different parameterizations are evaluated with metrics such as Mean Percent Error (MPE) and Mean Absolute Error (MAE), though the functions
can be easily edited to display different metrics such as predicted / true. Many of them also rely on specific folder and path naming 
conventions I developed and setting ```trace=True``` during the hls4ml conversion - see the PTQ and QAT folders 
for more details on how to implement this. 
   




