from misc import *
from sys import argv
from compiler import CompilationUnit, CompilationError

def parse_command_line_args(args: list[str]) -> tuple[str, str]:
  msg = 'Command syntax: bin/c_compiler -S [source-file.c] -o [dest-file.s]'
  expect_or(len(args) == 4, msg)

  s, source_path, o, output_path = tuple(args)

  expect_or(s == '-S', msg)
  expect_or(o == '-o', msg)

  return source_path, output_path

if __name__ == '__main__':
  from sys import version_info

  if version_info < (3, 10):
    exit('Please use a python version >=3.10')

  compiler = CompilationUnit(*parse_command_line_args(argv[1:]))
  try:
    asm = compiler.asm()
  except CompilationError as e:
    exit(repr(e))
  except Exception as e:
    print(f'Compiler crashed, probably you are testing a not implemented feature): {e.args}')
    raise

  with open(compiler.output_path, 'w') as output:
    output.write(asm)
