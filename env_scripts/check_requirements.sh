#!/bin/bash

REQ_IN="requirements.in"
REQ_TXT="requirements.txt"
TMP_REQ_TXT="temp_requirements.txt"

if ! command -v pip-compile &> /dev/null
then
    echo "pip-compile could not be found. Please install pip-tools."
    exit 1
fi

pip-compile --quiet --output-file "$TMP_REQ_TXT" "$REQ_IN"

if diff <(tail -n +7 "$TMP_REQ_TXT") <(tail -n +7 "$REQ_TXT") > /dev/null; then
    echo "The requirements.txt file is up-to-date."
    rm "$TMP_REQ_TXT"
    exit 0
else
    echo "The requirements.txt file is different from the compiled requirements.in."
    echo "You need to update the requirements.txt file. Here are the differences:"
    diff <(tail -n +7 "$REQ_TXT") <(tail -n +7 "$TMP_REQ_TXT")
    rm "$TMP_REQ_TXT"
    exit 1
fi
