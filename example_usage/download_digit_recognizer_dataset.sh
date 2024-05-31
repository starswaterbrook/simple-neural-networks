#!/bin/bash

download_file() {
    local url=$1
    local local_filename=$2
    
    curl -L -o "$local_filename" "$url"

    if [ $? -eq 0 ]; then
        echo "File downloaded successfully and saved as $local_filename"
    else
        echo "Failed to download file."
    fi
}

url="https://raw.githubusercontent.com/wehrley/Kaggle-Digit-Recognizer/master/train.csv"

local_filename="train.csv"

download_file "$url" "$local_filename"