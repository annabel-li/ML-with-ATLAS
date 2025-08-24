#ifndef ALGO_H
#define ALGO_H

#include "data.h"

#define NUM_LAYERS 2
#define NUM_VECS 12  // Aka fifo depth
#define MAX_ROWS 32 
#define MAX_COLS 32

#define ROWS0 32
#define COLS0 4
#define ROWS1 32
#define COLS1 32 //square weights matrix for layer 2 

#define LAST_ROW ROWS1 //the result of the matmul for 1 vector will have size (LAST_ROWS, 1)

#define NUM_FEATURES  COLS0
#define MAX_NODES MAX_ROWS 


// Enum for FSM states
enum fsm_state_t { IDLE, LOAD_MATRIX, LOAD_FIFO, LOAD_BIAS, COMPUTE_DENSE, READ_MATRIX, READ_FIFO, READ_BIAS, READ_RESULT, DONE };



extern "C" {
// implementation to be synthethised
void algo_main(
    volatile data_t *input_data,   // AXI4 burst source
    ctrl_t user_control,           // 1=load_matrix, 2=load_fifo, 3=load_bias, 4=compute, 5=read_matrix, 6=read_fifo, 7=read_bias, 8=read_result, 9=done
    ctrl_t layer_num,
    count_t weight_rows, count_t weight_cols, //user specifies the *actual* size of the weights matrix that's going to be loaded.  
    flag_t *busy,                   // Output: 1 while active, 0 when idle
    row_t read_row, col_t read_col, //both are used for both reading the weights matrix or reading from the dense layer results. 
    data_t *read_data,
    //data_t *element_sum_out, 
    count_t *nstates_visited_out,       // Output: number of states visited
    state_t *state_out,
    flag_t *done
);

}

// reference implementation for validation
void algo_main_ref(data_t *inData, data_t *outData, int nData);

#endif
