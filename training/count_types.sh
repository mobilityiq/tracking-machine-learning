#!/bin/bash

# Check if a file name is provided as an argument
if [ $# -eq 0 ]; then
  echo "Please provide the file name as an argument."
  exit 1
fi

file=$1

# Check if the file exists
if [ ! -f "$file" ]; then
  echo "File not found: $file"
  exit 1
fi

# Perform the count using awk
awk -F',' '{ count[$11]++ } END { for (type in count) print type, count[type] }' "$file"

