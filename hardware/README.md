## Hardware README.md - folder walk through  

### Folder structure 

```
   ├── hardware/
         └── python_qmodel/ 
         └── vitis_sample/ 
         - 'Getting started with Vitis & Vivado.pdf'
         - README.md
         - floats_for_hardware.py
```

### Content overview 

<p style="line-height: 2;">
   
#### 1) Scripts
- ```floats_for_hardware.py```: Converts Python floats to integer representations with specified fixed-point precision; used for input np.arrays. Official documentation about the ```ap_fixed<M, N>``` data type can be found on <a href="https://docs.amd.com/r/en-US/ug1399-vitis-hls/Overview-of-Arbitrary-Precision-Fixed-Point-Data-Types">AMD's website</a>.
   * Functions are meant for the PYNQ Jupyter interface <br>   
   * Python/NumPy has no native support for ap data types such as ap_uint<17> - this was a problem because our input stream was developed as an ap_uint<17> data type where the first 16 bits held data, and the leftmost bit held a '0' or a '1' corresponding to if this input vector signalled the end of an event or not (simulating detector EoEs)

      * Data processing pipeline I developed:
         1. Convert float to unsigned binary while storing the sign in another variable --> ```float_to_ubin()``` 
         2. Convert the unsigned binary to twos-complement form --> ```twos_complement()```
         3. Represent the binary as an integer --> ```bin_to_int()```
      * All functions are automatically called as a part of the main function, ```pckg_floats_for_fpga()```.

<br>

#### 2) Work samples
i. <b>```/python_qmodel/```</b> contains the ```DS_Model``` class I developed to simulate a neural network under hardware (ap_fixed) precisions.

Structure: 

```
   └── python_qmodel/
      - apy_ops.py
      - ds_utils.py
      - nn_utils.py
      - testDS.py
      - example_model.keras
```
   * ```apy_ops.py``` contains apy functions for neural network implementation 
   * ```ds_utils.py``` contains the definition of the quantized DS_Model class 
   * ```nn_utils.py``` contains the layer classes (Dense, KSum, and Regression)
   * ```testDS.py``` is the python test script where I tested an instance of a DS_Model
   * ```example_model.keras```: example model provided for testing the DS_Model class
       - Parameters: 64 nodes; 5 layers per Phi/F network; regression-only output; trained for ~100 epochs on -6 padded data

   <b> Deployment notes: </b>
   * DS_Model initialization requires the .keras file to the trained model and will automatically adjust to the layer shapes and number of layers 
       - If you have no trained models to test, you can use ```example_model.keras``` instead of training from scratch  
   * Model precision can be changed after initialization: the default is ```ap_fixed<16,6>``` for weights & biases and intermediate & output vectors. 
   * current ```Dense_Layer``` class performs a matrix multiplication between an input vector extended along 
   the column dimension with ones and a concatenated weights and biases matrix (instead of the usual V*W + B) due to the fact that APyFixedAccumulatorContext, which 
   allows for intermediate output quantizations that are not the optimized default, only works for inner products and matrix multiplications. The "old" method 
   of casting the sum of V*W + B is still in the script but commented out. If that is the route you decide to choose: 
       * Uncomment ```self.quantize_weights()``` and comment out ```self.process_wb()```
       * Uncomment the block under ```"old casting method"``` under Dense_Layer's ```forward_pass()``` method and comment out the block under 
       ```"concatenated wb method"```.
<br>

ii. <b>```/vitis_example/```</b> contains a sample of a finite state machine (FSM) I developed for testing on the FPGA along the way to crafting the full-fledged neural network.

Structure: 

```
   └── vitis_example/
      - algo.cpp
      - algo.h
      - data.h
      - tb_algo.cpp
      - hls_config.cfg
      - block_diagram.png
      - fsm5testing.ipynb
```

   * Vitis C++ files for synthesis: ```algo.cpp```, ```algo.h```, ```data.h```, ```tb_algo.cpp```, ```hls_config.cfg```
   * Vivado block diagram showing the IP block schematic: ```block_diagram.png```
   * Jupyter notebook for testing deployment on the FPGA: ```fsm5testing.ipynb```

#### 3) Tutorial Presentation
- ```Getting started with Vitis & Vivado.pdf```: teaches new users how to set up, connect, and run code on a PYNQ Z2 board with Vitis, Vivado, and PYNQ's Python API on JupyterLab
</p>
