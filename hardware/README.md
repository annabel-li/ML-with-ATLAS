# Hardware README.md - folder walk through  

## 1. Helper scripts 

```├── hardware/``` <br>
```    └── floats_for_hardware.py``` (Converts Python floats to integer representations with specified fixed-point precision; used for input np.arrays)

## 2. Work samples 
```├── hardware/```<br> 
```     │└── vitis_example/ ```<br>
```          └── algo.cpp``` <--- Vitis C++ files for synthesis<br>
```          └── algo.h```<br> 
```          └── data.h```<br>
```          └── tb_algo.cpp```<br>
```          └── hls_config.cfg```<br>
```          └── block_diagram.png``` <--- Vivado block diagram showing IP block schematic<br> 
```          └── fsm5testing.ipynb``` <--- Jupyter notebook for testing deployment on the FPGA 

## Sample workflow for Vitis & Vivado 

