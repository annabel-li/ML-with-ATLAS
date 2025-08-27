## Hardware README.md - folder walk through  

### Folder structure 

```
   ├── hardware/
        └── vitis_example/
            - algo.cpp
            - algo.h
            - data.h
            - tb_algo.cpp
            - hls_config.cfg
            - block_diagram.png
            - fsm5testing.ipynb
         - 'Getting started with Vitis & Vivado.pdf'
         - README.md
         - floats_for_hardware.py
```

### Content overview 

1) Scripts
- ``` floats_for_hardware.py```: Converts Python floats to integer representations with specified fixed-point precision; used for input np.arrays. Official documentation
  about the ```ap_fixed<M, N>``` data type can be found on AMD's website: https://docs.amd.com/r/en-US/ug1399-vitis-hls/Overview-of-Arbitrary-Precision-Fixed-Point-Data-Types.
     - Functions are meant for the PYNQ Jupyter interface
             - Python/NumPy has no native support for ap_uint<17> - this was a problem because our input stream was developed as an ap_uint<17> data type where the first                16 bits held data, and the leftmost bit held a '0' or a '1' corresponding to if this input vector signalled the end of an event or not (simulating detector                  EoEs)
             - Data processing pipeline I developed:
                   1) Convert float to unsigned binary while storing the sign in another variable --> ```float_to_ubin()```
                   2) Convert the unsigned binary to twos-complement form --> ```twos_complement()```
                   3) Fit to specified total and integer bit width --> ```fit_to_width()```
                   4) Represent the binary as an integer --> ```bin_to_int()```

2) Work samples
- ```/vitis_example/``` contains a sample of a finite state machine (FSM) I developed for testing on the FPGA along the way to crafting the full-fledged neural network.
   * Vitis C++ files for synthesis: ```algo.cpp```, ```algo.h```, ```data.h```, ```tb_algo.cpp```, ```hls_config.cfg```
   * Vivado block diagram showing the IP block schematic: ```block_diagram.png```
   * Jupyter notebook for testing deployment on the FPGA: ```fsm5testing.ipynb```

3) Tutorial Presentation
- ```Getting started with Vitis & Vivado.pdf```: teaches new users how to set up, connect, and run code on a PYNQ Z2 board with Vitis, Vivado, and PYNQ's Python API on JupyterLab

