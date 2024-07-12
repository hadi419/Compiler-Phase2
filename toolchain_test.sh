#!/bin/bash

# Author : James Nock (@Jpnock)
# Year   : 2023

set -euo pipefail

python3 src/entry.py -S "compiler_tests/_example/example.c" -o "compiler_tests/_example/example.s"
riscv64-unknown-elf-gcc -march=rv32imfd -mabi=ilp32d -o "compiler_tests/_example/example.out" "compiler_tests/_example/example.s" "compiler_tests/_example/example_driver.c"

python3 src/entry.py -S "compiler_tests/_example/example2.c" -o "compiler_tests/_example/example2.s"
riscv64-unknown-elf-gcc -march=rv32imfd -mabi=ilp32d -o "compiler_tests/_example/example2.out" "compiler_tests/_example/example2.s" "compiler_tests/_example/example2_driver.c"

set +e
spike pk "compiler_tests/_example/example.out" 
if [ $? -eq 0 ]; then
    echo "Test successful"
else
    echo "The simulator did not run correctly :("
fi
set -e

rm -f compiler_tests/_example/example.out
rm -f compiler_tests/_example/example.s

rm -f compiler_tests/_example/example2.out
rm -f compiler_tests/_example/example2.s
