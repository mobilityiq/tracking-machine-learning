#!/bin/bash

# Check if a directory path is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <root-directory>"
    exit 1
fi

# Traverse through the directory
find "$1" -type f -name "*.txt" | while read -r file; do
    # Check if the short version of the file doesn't exist
    if [[ ! -f "${file%.*}_short.txt" ]]; then
        # Create a short version using awk (For this example, let's take the first 100 lines, but you can adjust as needed)
        awk 'NR <= 100000' "$file" > "${file%.*}_short.txt"
        echo "Short version for $file created."
    else
        echo "Short version for $file already exists. Skipping."
    fi
done
