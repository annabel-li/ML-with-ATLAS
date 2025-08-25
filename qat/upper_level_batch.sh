model_nodes=(64)

start=$SECONDS

#this particular sweep was a 2D sweep in which integer bits for the weights and biases were kept constant; change it to suit your needs. 
for nodes in "${model_nodes[@]}"; do
    for v_full in {18..32}; do          #linux for loops are inclusive! 
        for wb_full in {12..27}; do 
            v_precision=("$v_full" 11)
            wb_precision=("$wb_full" 3)
            current_model="${nodes}"."${v_precision[0]}"."${v_precision[1]}".wb_"${wb_precision[0]}"."${wb_precision[1]}" 
            echo "Working on $current_model" 
            time python mass_convert.py -gpu 4 -n "$nodes" -v "${v_precision[@]}" -wb "${wb_precision[@]}" -ep 20 -modt bothSweep
            echo "Finished generating conversion outputs for $current_model"
        done
    done
done 
duration=$(( SECONDS - start ))
echo "Total time elapsed: $duration seconds"
