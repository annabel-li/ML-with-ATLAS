//#include <cstdio>
//#include <cassert>
#include "algo.h"
#include <list>
#include <iostream>

void load_tensor(data_t *tensor, int num_rows, int num_cols); 
void verify_tensor(data_t *tensor, int num_rows, int num_cols, char *tensor_name); 

int main() {
    //Output variables

    ctrl_t ctrl = 1; 
    flag_t done = 0;
    flag_t busy = 0;
    count_t nstates_visited_out = 0; 
    state_t state_out = IDLE;
    row_t row_to_read = 1; 
    col_t col_to_read = 1; 
    data_t output_data;  
    count_t layernum = 0; 
    //data_t intermediate_sum; 

    //tensors
    data_t weight_data[MAX_ROWS*MAX_COLS]; 
    count_t row_sizes[] = {ROWS0, ROWS1}; 
    count_t col_sizes[] = {COLS0, COLS1}; 
    data_t fifo_data[NUM_VECS*NUM_FEATURES]; 
    data_t bias_data[MAX_NODES*NUM_LAYERS]; 
    char names[][9] = {"weights1", "weights2"};

    int ctrl_out = ctrl&0x7; 

    printf("Loading the weights matrices...\n");
    for (int i = 0; i < NUM_LAYERS; i++) {
        //load the values into the 1D tensor (initialization) to get loaded into the internal 2D tensors in the fsm later... 
        load_tensor(weight_data, row_sizes[i], col_sizes[i]); 
        //verify_tensor(weight_data, row_sizes[i], col_sizes[i], names[i]); //optional
        ctrl = 1; 
        algo_main(weight_data, ctrl, i, row_sizes[i], col_sizes[i], &busy, row_to_read, col_to_read, &output_data, //&intermediate_sum, 
            &nstates_visited_out, &state_out, &done);
        //printf("Number of states visited: %d\n", nstates_visited_out&0xff);
        printf("--------------\n");
    }

    printf("Loading fifo...\n"); 
    ctrl = 2;
    load_tensor(fifo_data, NUM_VECS, NUM_FEATURES); 
    algo_main(fifo_data, ctrl, layernum, NUM_VECS, NUM_FEATURES, &busy,row_to_read, col_to_read, &output_data, //&intermediate_sum,
        &nstates_visited_out, &state_out, &done);    

    printf("Loading biases...\n"); 
    ctrl = 3;
    load_tensor(bias_data, NUM_LAYERS, MAX_NODES); 
    //verify_tensor(bias_data, NUM_LAYERS, MAX_NODES, "bias"); 

    for (int i = 0; i < NUM_LAYERS; i++) {
        algo_main(bias_data, ctrl, i, MAX_NODES, NUM_LAYERS, &busy,row_to_read, col_to_read, &output_data, //&intermediate_sum,
            &nstates_visited_out, &state_out, &done);
    }


    printf("Computing...\n"); 
    ctrl = 4;
    algo_main(bias_data, ctrl, layernum, MAX_NODES, NUM_LAYERS, &busy,row_to_read, col_to_read, &output_data, //&intermediate_sum,
        &nstates_visited_out, &state_out, &done);

    printf("Reading results...\n"); 
    ctrl=8; 
    for (int row = 0; row < NUM_VECS; row++) {
        printf("Result for vector %d:", row); 
        for (int col = 0; col < LAST_ROW; col++) {
            algo_main(bias_data, ctrl, layernum, MAX_NODES, NUM_LAYERS, &busy, row, col, &output_data, //&intermediate_sum,
                &nstates_visited_out, &state_out, &done);
            std::cout << output_data << " ";            
        }

        printf("\n-------\n"); 
    }

    return 0;
}




void load_tensor(data_t *tensor, int num_rows, int num_cols) {
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            tensor[i * num_cols + j] = (i + j)*(2*(i%2)-1);
        }
    }
}

void verify_tensor(data_t *tensor, int num_rows, int num_cols, char *tensor_name) {

    printf("\nChecking values in %s:\n", tensor_name); 

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            std::cout << tensor[i*num_cols+j] << " "; //this is how you can print negative numbers.
        }
        std::cout << "\n"; 
    }

}

