#!/bin/bash

# Check arguments
if [[ "$#" -ne 1 ]]; then
    echo "Usage: $0 <path_to_files>"
    exit 1
fi

# File paths
MAG_FILE="$1/Mag.txt"
ACC_FILE="$1/Acc.txt"
RESULT_FILE="./result.txt"

# Constants
BATCH_SIZE=6000  # 300 seconds * 20Hz
DOWNSAMPLED_SIZE=1200

# Initialize line counter
line_counter=1

# Ensure the result file is empty or create it
> "$RESULT_FILE"

while true; do

    initial_timestamp=$(sed -n "${line_counter}p" "$ACC_FILE" | awk '{print $1}')

    
    # Extract current batch from both files with downsampling
    mag_batch=$(sed -n "$line_counter,$((line_counter+BATCH_SIZE-1))p" "$MAG_FILE" | awk 'NR % 5 == 1')
    acc_batch=$(sed -n "$line_counter,$((line_counter+BATCH_SIZE-1))p" "$ACC_FILE" | awk 'NR % 5 == 1')

    # Debug statement to print the number of lines pulled for each file
    echo "Mag batch lines: $(echo "$mag_batch" | wc -l)"
    echo "Acc batch lines: $(echo "$acc_batch" | wc -l)"

    # Break the loop if there's no more data
    if [[ -z "$mag_batch" || -z "$acc_batch" ]]; then
        break
    fi

     

    # Prepare data for sending by interleaving mag and acc data
    combined_data=$(paste <(echo "$acc_batch") <(echo "$mag_batch" | awk '{print $2, $3, $4}') | awk '{print $1, $2, $3, $4, $5, $6, $7}')


    # Convert combined data to JSON
    json_data=$(echo "$combined_data" | awk 'BEGIN {print "["} {print "{\"timestamp\": " $1 ", \"x\": " $2 ", \"y\": " $3 ", \"z\": " $4 ", \"mx\": " $5 ", \"my\": " $6 ", \"mz\": " $7 "},"} END {print "]"}' | tr '\n' ' ' | sed 's/, ]/ ]/g')
    

    # Send data via curl and capture the response
    response=$(curl -X POST -H "Content-Type: application/json" -d "$json_data" http://192.168.18.200:8000/predict-bi-lstm)

     # Extract the final timestamp after the downsampling
    final_timestamp=$(echo "$acc_batch" | tail -n 1 | awk '{print $1}')

    echo "Initial Timestamp: $initial_timestamp"
    echo "Final Timestamp: $final_timestamp"
    # echo "Acc batch size after downsampling: $(echo "$acc_batch" | wc -l)"
    # echo "Mag batch size after downsampling: $(echo "$mag_batch" | wc -l)"

    # Append the results to the result file
    echo "$initial_timestamp,$final_timestamp,$response" >> "$RESULT_FILE"


    # Move to the next batch
    line_counter=$((line_counter + BATCH_SIZE))


done

echo "Processing complete."
