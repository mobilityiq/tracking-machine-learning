#!/bin/bash

# Check if the input file is provided as a command-line argument
if [ $# -lt 1 ]; then
  echo "Please provide the input file as a command-line argument."
  exit 1
fi

# Set the input file name
input_file="$1"

# Set the output file name
output_file="balanced_training.txt"

# Create a temporary file to store the mode counts
mode_counts_file="mode_counts.txt"

# Get the counts for each transportation mode
cut -d ',' -f 1 "$input_file" | sort | uniq -c > "$mode_counts_file"

# Find the mode with the lowest count
min_count=$(awk 'NR == 1 || $1 < min { min = $1; mode = $2 } END { print mode }' "$mode_counts_file")

# Get the minimum mode count
min_count_value=$(grep -w "$min_count" "$mode_counts_file" | awk '{print $1}')

# Set the desired line count for each mode
desired_count=$min_count_value

# Create a temporary file to store the shuffled lines
shuffled_file="shuffled_lines.txt"

# Create a new file with the desired number of lines for each mode
grep -w "$min_count" "$input_file" > "$shuffled_file"

# Loop through the other modes
grep -vw "$min_count" "$mode_counts_file" | while read -r count mode; do
  grep -w "$mode" "$input_file" | shuf -n "$desired_count" >> "$shuffled_file"
done

# Shuffle the lines in the temporary file
shuf "$shuffled_file" > "$output_file"

# Remove the temporary file
rm "$shuffled_file" "$mode_counts_file"

echo "New balanced training file created: $output_file"
