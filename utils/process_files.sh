#!/bin/bash

# Check if a path is provided
if [ -z "$1" ]; then
    echo "Please provide a path."
    exit 1
fi

# Check if the provided path exists
if [ ! -d "$1" ]; then
    echo "The provided path does not exist."
    exit 1
fi

# Find all .txt files recursively in the provided directory and its subdirectories
find "$1" -type f -name "*.txt" | while read -r file; do
    echo "Processing folder: $(dirname "$file")"
    echo "Working on file: $file"
    
    # For each file, replace multiple tabs with a single space
    awk 'BEGIN { OFS=" "; } { $1=$1; print }' "$file" > "${file}.tmp"
    # Overwrite the original file with the processed one
    mv "${file}.tmp" "$file"

    echo "Finished processing $file"
    echo "------------------------"
done

echo "Processing complete."
