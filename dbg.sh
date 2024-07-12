python3 src/entry.py -S "compiler_tests/_example/$1.c" -o "compiler_tests/_example/$1.s"

if [ $? -ne 0 ]; then
  echo "Could not compile"
  exit
fi

riscv64-unknown-elf-gcc -march=rv32imfd -mabi=ilp32d -c -o "compiler_tests/_example/$1.o" "compiler_tests/_example/$1.s"

if [ $? -eq 0 ]; then
  echo "Test successful"
else
  echo "Could not assemble"
fi

rm -f "compiler_tests/_example/$1.o"
rm -f "compiler_tests/_example/$1.s"
