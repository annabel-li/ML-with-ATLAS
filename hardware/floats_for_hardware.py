import numpy as np 

#these functions are for the Jupyter (Python) interface -> takes floats and converts them to fixed-point representations and then 
#repackages as integers to feed into an FPGA in matrix form. 

<<<<<<< Updated upstream
"""NOTE: If using these functions, the only function you need to call is pckg_float_for_fpga(). 
It automatically calls the other functions when running."""
=======
#the only function you need to call is pckg_float_for_fpga(). 
>>>>>>> Stashed changes


def get_frac_bits(
    in_float: float, 
    round_factor: int=8, #number of decimal places to round to. 
):
    decimal_part, int_part = np.modf(in_float)
    decimal_part = np.abs(np.round(decimal_part, round_factor))
    dec_string = str(decimal_part)[2:] #omit the '0.'

    if decimal_part == 0: 
        frac_bits = 0 
    else: 
        power = len(dec_string) 
        frac_bits = math.log(10**power, 2) #base 2 log. 
        frac_bits = int(frac_bits) + 1 #round up 

    return frac_bits  
 
def get_int_bits(
    in_float: float
): 
    int_part = int(in_float)
    return len(bin(int_part)) - 2 #get rid of the '0b' prefix 

def decimal_to_bin(
    decimal_float: float, 
    user_frac_bits: int
): 

    before_decimal = 0 
    decimal_str = []
    
    while(before_decimal != 1 and (len(decimal_str) < user_frac_bits)): 

        decimal_float = decimal_float * 2 
        decimal_float, int_part = np.modf(decimal_float) 
        decimal_str.append(str(int(int_part)))

    decimal_str = "".join(decimal_str) 

    return decimal_str     

"""
in hardware, if the number of bits for the integer part is not enough but there's the right
number for the decimal, the computer doesn't care; it just sees the binary string and will 
insert the decimal point according to the fixed precision you give it. Thus, if you have 
something like ap_fixed<4,2> for 6.25, even though the decimal part *could* be represented accurately 
with 2 bits, it won't because the integer part will overflow into it. That's why you should just concatenate 
the strings here. 
"""

#float to unsigned binary. Returns a list in the form of [unsigned binary representation, sign]
def float_to_ubin(
    in_float: float,
    user_tot_bits: int, #user-specified full bit width. 
    user_int_bits: int, #user-specified int width for ap_fixed conversion. 
    round_factor: int=8, 
    round_num: bool=True
): 

    user_frac_bits = user_tot_bits - user_int_bits
    
    if in_float < 0: #store the sign. 
        sign = "-" 
    elif in_float > 0: 
        sign = "+" #me being explicit and also consistent 
    else: 
        sign = " "

    in_float = np.abs(in_float) 
    decimal_part, int_part = np.modf(in_float) 
    #print(f"Int: {int_part}, decimal: {decimal_part}")
    
    if round_num: 
    #round the float to the specified decimal place 
        decimal_part = np.round(decimal_part, round_factor)
        #print(decimal_part)
        in_float = int_part + decimal_part

    #from https://www.geeksforgeeks.org/python/python-program-to-convert-floating-to-binary/: 
    #IEEE 754 binary representation
    #binary_rep = np.binary_repr(np.float32(in_float).view(np.int32), width=32)
    #This is precise, but not exactly ideal for this particular context.  

    #frac_bits = get_frac_bits(in_float) 
    #print("Frac bits:", frac_bits) 
    #scaled = int(in_float * (2**user_frac_bits)) 
    #binary_string = format(scaled, 'b') #formats the scaled number as binary 

    int_part = bin(int(int_part))[2:].zfill(user_int_bits)
    decimal_part = decimal_to_bin(decimal_part, user_frac_bits)

    binary_string = int_part + decimal_part

    bin_list = [binary_string, sign] #this returns the bin string with extraneous zeros.
      
    return bin_list 


def invert_bits(
    bin_str: str
): 
    
    if '.' in bin_str: 
        bin_str = bin_str.replace('.', "")
        
    bin_str = list(bin_str)
    
    for i in range(len(bin_str)): 
        if bin_str[i] == '0':
            bin_str[i] = '1' 
        elif bin_str[i] == '1': 
            bin_str[i] = '0' 
        else: 
            raise ValueError("Unrecognized character in binary string.")
            
    bin_str = "".join(bin_str)
    
    return bin_str

def add_one_to_bin_str(
    bin_str: str
): 

    stop_carry = False
    length = len(bin_str)  
    bin_str = list(bin_str)
    
    #deal with the easy case first: where the last value is 0. 
    if bin_str[length - 1] == '0': 
        bin_str[length - 1] = '1'

    else: 
        
        for i in np.flip(np.arange(1, length)):          
            if bin_str[i] == '1' and stop_carry == False: 
                bin_str[i] = '0' 

                if bin_str[i - 1] == '1':
                    bin_str[i - 1] = '0'

                else:
                    bin_str[i - 1] = '1'
                    stop_carry = True #stop carrying over. 
        
        if stop_carry == False: #once you get to the end... 
            bin_str = '1' + bin_str #extend by 1. 
            
    bin_str="".join(bin_str)
    return bin_str 


def twos_complement(
    bin_list: list
): 
    ubin_str, sign = bin_list
    nbits = len(ubin_str) + 1

    if sign == '+': 
        sbin_str = '0' + ubin_str #implicitly converts to string. 
        
    elif sign == '-': 
        #get the binary string of the positive number -> invert all the bits -> add 1. recall that bin_list[0] is the binary of the unsigned number. 
        inverted = invert_bits(ubin_str) 
        #print("inverted: ", inverted)
        added_one = add_one_to_bin_str(inverted)
        #print("Added one: ", added_one)
        sbin_str = added_one

    else: 
        sbin_str = ubin_str #cuz it's just all zeros. 

    return sbin_str


#The first bit is the signed bit in twos complement notation -> will need an extra bit allocated to the integer portion 
#twos_complement function should be called before this one. 
#this function doesn't work because it takes in int_bits again even when the first original number is already 
#at the specified bit width... this is an obsolete function no longer fit for this workflow. 
"""
def fit_to_width(
    bin_str: str,  
    int_bits: int, 
    frac_bits: int, 
    original_float: float
): 
    
    target_width = int_bits + frac_bits 
    current_width = len(bin_str) 

    if int_bits < 0 or frac_bits < 0: 
        raise ValueError("Number of bits cannot be negative!")

    #left pad if int_bits > int bits needed 
    if int_bits > get_int_bits(original_float): 
        if original_float < 0: #pad with 1 if negative
            for i in range(int_bits - get_int_bits(original_float)): 
                bin_str = '1' + bin_str  
        else: #pad with 0 if positive
            for i in range(int_bits - get_int_bits(original_float)):
                bin_str = '0' + bin_str
        current_width = len(bin_str) #update current width.
        print(bin_str)

    #if current width is too small, right-pad with zeroes 
    if current_width < target_width: 
        
        for i in range(target_width - current_width):
            bin_str = bin_str + '0'
        print(bin_str)

    if frac_bits == 0 or int_bits == 0: 
        full_num = bin_str + '.' 
    elif int_bits == 0: 
        full_num = '.' + bin_str 
    else:    
        int_part = bin_str[:int_bits] 
        print("int part:", int_part) 
        frac_part = bin_str[int_bits:int_bits+frac_bits]
        print("frac part:", frac_part) 
        full_num = int_part + '.' + frac_part 

    return full_num #signed binary representation split into int + frac form 
"""

def bin_to_int(
    bin_str: str
): 
    
    if '.' in bin_str: 
        bin_str = bin_str.replace('.', "")

    int_rep = int(bin_str, 2)

    return int_rep 


def attach_flag(
    bin_string: str, 
    flag: int, 
    twos_complement: bool=True #set to true if you passed it through the twos complement function before this. 
): 
    
    if twos_complement: 
        if bin_string[0] == '0':  #get rid of the automatic zero padding for positive numbers
            return str(flag) + bin_string[1:]
        
        else:  # negative case
            return str(flag) + bin_string[:-1] #make negative numbers the same length as positive numbers.

    else: 
        return str(flag) + bin_string

def pckg_float_for_fpga(
    num: float, 
    total_width: int, 
    int_width: int, 
    flag: int=None
):
    
    ubin_num = float_to_ubin(num, total_width, int_width)
    twocomp_num = twos_complement(ubin_num)
    #fixed_rep = fit_to_width(twocomp_num, int_width, total_width-int_width, num)
    
    if flag is not None: 
<<<<<<< Updated upstream
        addflag = attach_flag(twocomp_num, flag)
        print("With flag: ", addflag)
        twocomp_num = addflag
        #fixed_rep = addflag
        
=======
        #addflag = attach_flag(fixed_rep, flag)
        addflag = attach_flag(twocomp_num, flag) 
        print("With flag: ", addflag)
        twocomp_num = addflag
        
    #final_int = bin_to_int(fixed_rep)
    print(twocomp_num) 
>>>>>>> Stashed changes
    final_int = bin_to_int(twocomp_num)

    return final_int















