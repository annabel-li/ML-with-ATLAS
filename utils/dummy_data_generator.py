import numpy as np 

def dummy_data_gen(
    num_samples: int=100000, 
    vmin: int=4, 
    vmax: int=128, 
    nfeatures: int=4, 
    pad: int=-6
): 
    
    data_list = [] 
    regression_targets = []

    if vmin > vmax: 
        raise ValueError("User-specified vmin is greater than vmax.")

    for i in range(num_samples): 

        #fill with numbers between 0 and 1 up to a randomly generated "actual cluster length"
        len_real_data = np.random.randint(vmin, vmax)
        real_data = np.random.rand(len_real_data, nfeatures) 

        #pad up to vmax with the padded value 
        full_data = np.pad(real_data[0:vmax], [(0, vmax-len_real_data), (0, 0)], 'constant', constant_values=pad)

        #append to data_list and append a fake target to "targets."
        data_list.append(full_data)
        regression_targets.append(np.random.rand(1))

    return np.array(data_list), np.array(regression_targets) #return as a tuple with 2 np arrays 

