## Pruning folder introduction 

```
|-- prune/
    - README.md
    - analyze_prune.py 
    - train_model_prune.py 
```

Using tensorflow's [built-in pruning library](https://www.tensorflow.org/model_optimization/guide/pruning/comprehensive_guide.md), we can create models 
that are significantly smaller in size with a minimal loss in accuracy. To do so, weights with 'lower signifcance' are zeroed out until the user-set sparsity, 
expressed as a decimal < 1, is reached. ('Low significance' is determined by a threshold value, which is also determined from the specified sparsity.) 
The reason why accuracy losses can be minimal is that some weights "contribute less" to the final output than others; 
pruning identifies these weights and removes them from the network by zeroing them out. 

To train a model with pruning, it is highly recommended to load the weights from a properly pre-trained model first (eg. trained on a large dataset 
for around 100 epochs; the ```trainModel.py``` script under ```ptq/``` is meant to serve this purpose.) This decreases training time and increases 
accuracy. 

Tensorflow also has built-in training logs on Tensorboard that can be viewed during training, in which users can see the model's sparsity at any given step. 
To read these logs: 
1. Open a new terminal window 
2. Enter the container/virtual environment where your pruning training is running 
3. Run: ```tensorboard --logdir=<path_to_log_folder>``` <br>
   **If you are in an ssh environment, in a new tab in your terminal, run: ```ssh -L 6006:localhost:6006<your_username>@ssh_env_name```
4. Open the given link in your browser (should be something like https://localhost:6006/)

### Current progress 

So far, I have tried the following sparsities: 
- Target: 0.3; training epochs: 33 (number of epochs was unspecified; the original target sparsity of 0.5 was never reached because early stopping kicked in.
After this, I got rid of the early stopping callback.)
- Target: 0.5; training epochs: 20 (+3 'warm up' epochs)
- Target: 0.75; training epochs: 35 (+3 'warm up' epochs)
After testing with 1000 input samples, the error (pred/true) between the full Keras model and the pruned models varies by only ~1% (original Keras pred/true 
= 1.037, 75% sparsity pred/true = 1.041.) Thus pruning would be a promising avenue to investigate further. 


### More reading 

For more information on pruning, I recommend these resources: 
- https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras 
- https://www.tensorflow.org/model_optimization/guide/pruning/comprehensive_guide.md 
- https://github.com/fastmachinelearning/hls4ml-tutorial/blob/main/part3_compression.ipynb 
- https://wandb.ai/authors/pruning/reports/Diving-Into-Model-Pruning-in-Deep-Learning--VmlldzoxMzcyMDg  (theory behind pruning)



