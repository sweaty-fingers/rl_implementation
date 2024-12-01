#!/bin/bash

# Find all files with .sh extension in the scripts directory and its subdirectories
# and add execute permission to them
run() {
    echo "Execution permissions have been set for all script files in the scripts directory and its subdirectories."
    find scripts -type f -name "*.sh" -exec chmod +x {} \;
}  

case $1 in
    run)
        run
        ;;
    *)
        echo "Usage: $0 {run}"
        ;;
esac