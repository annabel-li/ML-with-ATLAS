#ifndef ALGO_DATA_H
#define ALGO_DATA_H

#include "ap_int.h"
#include "ap_fixed.h"
#include "ap_axi_sdata.h"
#include <hls_stream.h>
#include <hls_vector.h>

typedef ap_int<32> data_t;
typedef ap_uint<7> row_t;
typedef ap_uint<7> col_t;
typedef ap_uint<1> flag_t;
typedef ap_uint<3> state_t;
typedef ap_uint<5> ctrl_t;
typedef ap_uint<8> count_t; 

#endif