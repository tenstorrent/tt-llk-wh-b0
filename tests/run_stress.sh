#!/bin/bash

# Check if CHIP_ARCH is set to wormhole or blackhole
if [[ "$CHIP_ARCH" = "wormhole" ]]; then
    /home/software/syseng/wh/tt-smi -wr 0
elif [[ "$CHIP_ARCH" = "blackhole" ]]; then
    tt-smi -r 0
else
    echo "No architecture detected"
fi

# Function to display usage instructions
usage() {
    echo "Usage: $0 --repeat <number_of_repeats> --test <test_name>"
    exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repeat)
            repeat_count="$2"
            shift 2
            ;;
        --test)
            test_name="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

# Ensure both parameters are provided
if [[ -z "$repeat_count" || -z "$test_name" ]]; then
    usage
fi

pass_count=0
fail_count=0

# Run the test for the specified number of iterations
for i in $(seq 1 "$repeat_count"); do
    pytest "$test_name" > /dev/null 2>&1
    result=$?
    
    if [ $result -eq 0 ]; then
        ((pass_count++))
    else
        ((fail_count++))
    fi
done

# Report results
echo "Passed: $pass_count"
echo "Failed: $fail_count"
