#!/bin/bash

ls

# Directory where the original files are located
SOURCE_DIR="data/binance"

# Directory where the USD pair files should be copied
DEST_DIR="data/binance/usd/"

# Create the destination directory if it does not exist
mkdir -p "$DEST_DIR"

# Copy all files that end with 'usd.csv' from source to destination directory
find "$SOURCE_DIR" -type f -iname "*usdt.parquet" -exec cp {} "$DEST_DIR" \;

echo "Copy completed."