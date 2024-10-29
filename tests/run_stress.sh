#!/bin/bash

# initial reset of board 0
/home/software/syseng/wh/tt-smi -wr 0

pass_count=0
fail_count=0

test_name="test_eltwise.py"

for i in {1..100}; do
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
