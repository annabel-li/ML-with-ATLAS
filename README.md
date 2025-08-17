# Optimizing software to hardware conversions of ML models for the ATLAS L0 trigger system
## Goals

The goal of this project was to investigate how different parameterizations of a machine learning model impacted its ability to perform on hardware. I trained and quantized models using Keras, then converted them into High Level Synthesis (HLS)-ready form with the <b>hls4ml</b> library. Strategies explored included pruning, Quantization-Aware Training (QAT), and Post-Training Quantization (PTQ). 


## About the project   

With the Large Hadron Collider (LHC)’s forthcoming High-Luminosity upgrade, the number of collisions per bunch crossing is expected to more than triple. This places more demand on the hardware trigger system, which must improve its accuracy, efficiency, and resource usage to take advantage of the greater opportunities to capture rare high-energy physics events. 

Deep learning models such as the [DeepSets](https://arxiv.org/pdf/2402.01876), developed specifically for such contexts, perform well at cluster energy regression and pion type classification. This makes them a strong candidate for deployment on the trigger system’s FPGAs. However, a model’s performance in software does not always translate directly to hardware. Differences in numerical representation, memory, and timing constraints can significantly affect accuracy.

In software, numbers are represented in floating point (eg. “float32”). On hardware, these are converted to fixed-point representations, where the quantity of bits for the integer and decimal parts of each number is preset. This process, called <b>quantization,</b> reduces memory and computational requirements, but also limits the range of numbers that can be properly represented by the system, increasing the risk of overflow or underflow. 

To tackle this, machine learning researchers use one of two strategies: 

<ul> 
  <li><b>Post-Training Quantization (PTQ):</b> weights and biases are quantized after training </li>
  <li><b>Quantization-Aware Training (QAT):</b> quantization effects are simulated during training, allowing the model to learn parameters that are less sensitive to reduced precision. </li>
</ul>

[Hls4ml](https://fastmachinelearning.org/hls4ml/) is a particularly useful library for testing this as it automates the process of converting Python-based ML models into HLS-ready form for FPGA synthesis. When combined with tools such as QKeras, it provides an efficient way to test how different model parameterizations impact the balance between hardware efficiency and accuracy retention. 

My research this term investigated how such design choices - such as layer size, quantization strategy, and precision settings - affected the accuracy gap between the original Keras model and its hls4ml implementation. These findings guide the optimization of the DeepSets for deployment in ATLAS’s hardware trigger system, ensuring fast, accurate, and resource-conscious real-time data analysis. 

## References and Resources 

My implementation of the DeepSets model was based on the work done by the authors of the "Ultrafast Jet Classification on FPGAs for the HL-LHC". You can find their repository <a href="https://github.com/fastmachinelearning/l1-jet-id/tree/main/fast_jetclass/deepsets">here.</a>  

I also found these <a href = "https://github.com/fastmachinelearning/hls4ml-tutorial/blob/main/part1_getting_started.ipynb">Hls4ml tutorials</a> extremely helpful, as they cover everything from basic model conversion to training with QKeras and Pruning. If you would also like to implement custom layers with Hls4ml, check out their page on [extension API.](https://fastmachinelearning.org/hls4ml/advanced/extension.html#extension-api)
