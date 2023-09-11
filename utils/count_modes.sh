#!/bin/bash

dir=$1

translate_mode() {
  case $1 in
    0) echo "UNKNOWN" ;;
    1) echo "STATIONARY" ;;
    "walking") echo "walking" ;;
    "running") echo "running" ;;
    "cycling") echo "cycling" ;;
    "drivint") echo "driving" ;;
    6) echo "bus" ;;
    "train") echo "train" ;;
    8) echo "SUBWAY" ;;
    *) echo "UNKNOWN" ;;
  esac
}

plot_graph_for_file() {
    file=$1
    echo "Processing file: $file"
    
    awk '{print $10}' $file | sort | uniq -c | while read count mode; do
        translated_mode=$(translate_mode $mode)
        echo "$translated_mode $count"
    done > graph_data.txt

    # Dynamically generate the color sequence
    colors=("red" "blue" "green" "magenta" "yellow" "cyan" "black")
    num_modes=$(awk '{print $1}' graph_data.txt | wc -l)
    color_sequence=()
    for (( i=0; i<$num_modes; i++ )); do
        color_sequence+=("${colors[$i % ${#colors[@]}]}")
    done

    termgraph graph_data.txt --color ${color_sequence[@]}
    echo "----------------------------------"
}

# Iterate over subdirectories and look for .....
find $dir -type f -name "training-3.0.csv" | while read file; do
    plot_graph_for_file $file
done
