#include "algo.h"

void load_weights(
    volatile data_t *inputs, //1D flattened tensor with size num_rows * num_cols  
    data_t matrix[NUM_LAYERS][MAX_ROWS][MAX_COLS],
    count_t layer_num, 
    count_t actual_rows, 
    count_t actual_cols
); 

void load_bias(
    volatile data_t *inputs, 
    data_t matrix[NUM_LAYERS][MAX_NODES],
    count_t layer_num, 
    count_t num_nodes_current, 
    data_t nodelist[NUM_LAYERS]
);

void load_fifo(
    volatile data_t *inputs, 
    data_t matrix[NUM_VECS][NUM_FEATURES],
    count_t num_vecs
); 

void dense(
    count_t layer_num,  
    data_t *input_vec, 
    data_t weights[NUM_LAYERS][MAX_ROWS][MAX_COLS], 
    row_t weights_rows, 
    col_t weights_cols, 
    data_t bias_tensor[NUM_LAYERS][MAX_NODES], 
    data_t *output_vec
); 

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
)
{

    //IMPORTANT: Bias, results, and fifo are stored row-wise, even though the matmul assumes the input vector is a column vector!

#pragma HLS INTERFACE m_axi     port=input_data offset=slave bundle=gmem depth= NUM_LAYERS*MAX_ROWS*MAX_COLS //allocate enough memory to satisfy the largest matrix
#pragma HLS INTERFACE s_axilite port=input_data bundle=control
#pragma HLS INTERFACE s_axilite port = user_control bundle = control
#pragma HLS INTERFACE s_axilite port = layer_num bundle = control
#pragma HLS INTERFACE s_axilite port = weight_rows bundle = control
#pragma HLS INTERFACE s_axilite port = weight_cols bundle = control
#pragma HLS INTERFACE s_axilite port = busy bundle = control
#pragma HLS INTERFACE s_axilite port = state_out bundle = control
#pragma HLS INTERFACE s_axilite port=read_row    bundle=control
#pragma HLS INTERFACE s_axilite port=read_col    bundle=control
#pragma HLS INTERFACE s_axilite port=read_data   bundle=control
//#pragma HLS INTERFACE s_axilite port=element_sum_out  bundle=control
#pragma HLS INTERFACE s_axilite port = nstates_visited_out bundle = control
#pragma HLS INTERFACE s_axilite port=done        bundle=control
#pragma HLS INTERFACE s_axilite port=return      bundle=control

    //static data_t weights0[ROWS1*COLS1];
    //static data_t **weights0; //doubel pointers are not synthesizable HLS : https://docs.amd.com/r/en-US/ug1399-vitis-hls/Mapping-Software-Arrays-to-Hardware-Memory

    //NN architecture 
    static data_t weights[NUM_LAYERS][MAX_ROWS][MAX_COLS]; 
    static data_t biases[NUM_LAYERS][MAX_NODES];

    //data storage 
    static data_t vec_storage[NUM_VECS][NUM_FEATURES]; 
    static data_t results[NUM_VECS][LAST_ROW]; //Matrix to store FINAL matmul results 
    static data_t num_nodes[] = {ROWS0, ROWS1}; //stores the number of nodes per layer. 

    static data_t debugger; 

 //old  #pragma HLS RESOURCE variable=bram core=RAM_2P_BRAM

    #pragma HLS BIND_STORAGE variable=weights type=RAM_2P impl=BRAM
    #pragma HLS ARRAY_RESHAPE variable=weights complete dim=3
    #pragma HLS ARRAY_PARTITION variable=weights type=block factor=MAX_ROWS*MAX_COLS dim=1 //splits the matrix up into blocks of size MAX_ROWS x MAX_COLS so you can access different layers at the same time


    #pragma HLS BIND_STORAGE variable=biases type=RAM_2P impl=BRAM
    #pragma HLS ARRAY_RESHAPE variable=biases complete dim=2

    #pragma HLS BIND_STORAGE variable=vec_storage type=RAM_2P impl=BRAM
    #pragma HLS ARRAY_RESHAPE variable=vec_storage complete dim=2

    #pragma HLS BIND_STORAGE variable=results type=RAM_2P impl=BRAM
    #pragma HLS ARRAY_RESHAPE variable=results complete dim=2


    // FSM state variables
    static fsm_state_t state = IDLE; //static means the variable's value persists between function calls 
    static count_t nstates_visited = 0;   
    static count_t vec_to_read = 0;
    //static data_t element_sum; 

    data_t layer1_output[MAX_NODES];
    data_t layer2_output[MAX_NODES];

    *done = 0;  // Initialize to not done
    *busy = (state != IDLE);

    do {
        switch (state) {
        case IDLE:
                // Wait for ap_start (triggered by host via AXI Lite "start")
                if (user_control == 1)
                {
                    state = LOAD_MATRIX;
                }
                else if (user_control == 2)
                {
                    state = LOAD_FIFO;
                }
                else if (user_control == 3)
                {
                    state = LOAD_BIAS;
                }
                else if (user_control == 4)
                {
                    state = COMPUTE_DENSE;
                }
                else if (user_control == 5)
                {
                    state = READ_MATRIX;
                }
                else if (user_control == 6)
                {
                    state = READ_FIFO;
                }
                else if (user_control == 7)
                {
                    state = READ_BIAS;
                }
                else if (user_control == 8)
                {
                    state = READ_RESULT;
                }
                else if (user_control == 9) 
                {
                    state = DONE;
                } 
                else {
                    // Handle unexpected control values
                    state = IDLE; // Stay in IDLE state
                    *done = 1; // Return from command
                }
            break;

        case LOAD_MATRIX: 
        //must call NUM_LAYER times to load the weight matrices and pass in layer_num, weight_rows and weight_cols as user-specified input, where weight_rows and weight_cols are the actual sizes 
        //of the weights matrix for that specific layer 

            load_weights(input_data, weights, layer_num, weight_rows, weight_cols); 
            
            state = DONE;
            break;

        case LOAD_FIFO:
            
            load_fifo(input_data, vec_storage, NUM_VECS); 

            state = DONE; 
            break;

        case LOAD_BIAS: //loads one layer at a time

            load_bias(input_data, biases, layer_num, weight_rows, num_nodes); 
  
            state = DONE; 
            break;

        case COMPUTE_DENSE:
            
            //compute matrix-vector multiplication for the vec_to_read-th vector in the vec_storage (sequential order) & store the result 

            for (vec_to_read = 0; vec_to_read < NUM_VECS; vec_to_read++) {

                dense(0, vec_storage[vec_to_read], weights, ROWS0, COLS0, biases, layer1_output); //first layer. 
                dense(1, layer1_output, weights, ROWS1, COLS1, biases, layer2_output); //second layer. 

                //copy layer output to results matrix. STORES THEM ROW-WISE!!
                for (int i = 0; i < LAST_ROW; i++) {
                    results[vec_to_read][i] = layer2_output[i]; 
                }
            }

            state = DONE;         
            break;

        case READ_MATRIX: //to inspect the weights matrix to ensure it has been loaded correctly 
            
            read_row = read_row % weight_rows; 
            read_col = read_col % weight_cols;
            layer_num = layer_num % NUM_LAYERS; 

            *read_data = weights[layer_num][read_row][read_col];
            state = DONE; 

            break;

        case READ_FIFO: //to inspect the fifo to ensure it has been loaded correctly 
            
            read_row = read_row % NUM_VECS; 
            read_col = read_col % NUM_FEATURES;           

            *read_data =vec_storage[read_row][read_col];

            state = DONE; 

            break;   

        case READ_BIAS: //to inspect the bias matrix to ensure it has been loaded correctly 
            
            read_row = read_row % NUM_LAYERS; 
            read_col = read_col % MAX_NODES;           

            *read_data =biases[read_row][read_col];

            state = DONE; 

            break;    

        case READ_RESULT: //to read back, one element at a time, the result of the matmul 

            read_row = read_row % NUM_VECS; 
            read_col = read_col % LAST_ROW;         

            *read_data =results[read_row][read_col];

            state = DONE; 

            break;       

        case DONE:
            *done = 1; // One-cycle pulse
            state = IDLE;                  // Update current FSM state
            break;              // Exit the FSM
        }

        nstates_visited++;

        *nstates_visited_out = nstates_visited; // Output number of states visited
        *state_out = state; 

    } while (!*done); // Loop until done

}


void load_weights(
    volatile data_t *inputs, //1D flattened tensor with size num_rows * num_cols  
    data_t matrix[NUM_LAYERS][MAX_ROWS][MAX_COLS],
    count_t layer_num, 
    count_t actual_rows, 
    count_t actual_cols
) {
    for (int i = 0; i < actual_rows; i++) {
        for (int j = 0; j < actual_cols; j++)
            matrix[layer_num][i][j] = inputs[i * actual_cols + j]; 
    }
}

void load_bias(
    volatile data_t *inputs, 
    data_t matrix[NUM_LAYERS][MAX_NODES],
    count_t layer_num, 
    count_t num_nodes_current, 
    data_t nodelist[NUM_LAYERS]
) {

    data_t offset = 0; 

    if (layer_num > 0) {
        for (int i = 0; i < layer_num; i++) {
            offset += nodelist[i]; 
        }
    }

    for (int i = 0; i < num_nodes_current; i++) {
        matrix[layer_num][i] = inputs[offset + i];  
    }
}

void load_fifo(
    volatile data_t *inputs, //1D flattened tensor with size num_rows * num_cols  
    data_t matrix[NUM_VECS][NUM_FEATURES],
    count_t num_vecs
) {
    for (int i = 0; i < num_vecs; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            matrix[i][j] = inputs[i * NUM_FEATURES + j]; 
        }
        
    }
}


void dense(
    count_t layer_num,  
    data_t *input_vec, 
    data_t weights[NUM_LAYERS][MAX_ROWS][MAX_COLS], 
    row_t weights_rows, 
    col_t weights_cols, 
    data_t bias_tensor[NUM_LAYERS][MAX_NODES], 
    data_t *output_vec
) {

    data_t element_sum;

    for (int i = 0; i < weights_rows; i++) {
        element_sum = bias_tensor[layer_num][i];

        for (int j = 0; j < weights_cols; j++) {
            element_sum += input_vec[j]*weights[layer_num][i][j]; 
        } 

        output_vec[i] = element_sum;
        
        //simulate the RELU activation function
        if (output_vec[i] < 0) { 
            output_vec[i] = 0;  
        }
    }

}


