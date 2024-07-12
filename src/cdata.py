from typing import Any

BINARY_OPERATORS = {
  '+': 'add', '-': 'sub',
  '*': 'mul', '/': 'div', '%': 'rem',
  '^': 'xor', '&': 'and', '|': 'or',
}

COMPLEX_BIN_OPERATORS = [
  '==', '!=', '<', '<=', '>', '>=',
  '<<', '>>'
]

LITERAL_CTYPES = [
  'lit_int'
]

VALUE_REGISTERS = [
  'a0', 'a1', 'a2', 'a3',
  'a4', 'a5', 'a6', 'a7'
]

RETURN_REG = VALUE_REGISTERS[0]
RETURN_ADDR_REG = 'ra'

LOAD_INSTRUCTIONS = [
  'flw', 'fld', # for floating points
  'lb', 'lh', 'lw', 'ld' # for integers
]

STORE_INSTRUCTIONS = [
  'fsw', 'fsd', # for floating points
  'sb', 'sh', 'sw', 'sd' # for integers
]

# this is useful to correctly output the stack size
# when early return is emitted
# otherwise it would contain an outdated value of
# the real stacksize one
STACKSIZE_PLACEHOLDER = '%STACK_SIZE%'

assert len(LOAD_INSTRUCTIONS) == len(STORE_INSTRUCTIONS)

def store_to_load_instruction(opname: str) -> str:
  assert opname in LOAD_INSTRUCTIONS

  return STORE_INSTRUCTIONS[LOAD_INSTRUCTIONS.index(opname)]

class CType:
  def __init__(self, tag: str, info: Any = None):
    self.tag = tag
    self.info = info

  def calculate_size(self) -> int:
    match self.tag:
      case 'void':
        return 0

      case 'int':
        return self.info

      case _:
        raise NotImplementedError(self.tag)

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, CType):
      return False

    if self.tag == other.tag:
      return self.info == other.info

    if 'lit' in [self.tag, other.tag]:
      return (
        self.tag == other.info or
        other.tag == self.info
      )

    return False

  def __repr__(self) -> str:
    if self.tag == 'lit':
      return f'literal {self.info}'

    return self.tag

class AssemblyInstr:
  def __init__(self, opname: str, args: list[Any]) -> None:
    self.opname = opname
    self.args = args

  @property
  def is_instr(self) -> bool:
    return True

  def to_string(self, indent: str) -> str:
    return f'{indent}{repr(self)}'

  def __eq__(self, other: object) -> bool:
    return (
      isinstance(other, AssemblyInstr) and
      self.opname == other.opname and
      self.args == other.args
    )

  def __repr__(self) -> str:
    args = ', '.join(map(str, self.args))

    return f'{self.opname} {args}'

class AssemblyLabel:
  def __init__(self, name: str) -> None:
    self.name = name

  @property
  def is_instr(self) -> bool:
    return False

  def to_string(self, indent: str) -> str:
    indent = indent[:-2]

    return f'\n{indent}{repr(self)}'

  def __repr__(self) -> str:
    return f'{self.name}:'

class AssemblyFn:
  def __init__(self, name: str) -> None:
    self.name = name
    self.stacksize = 4
    self.body: list[AssemblyInstr | AssemblyLabel] = []

  @property
  def is_terminated(self) -> bool:
    return (
      len(self.body) > 0 and
      self.last == AssemblyInstr('jr', [RETURN_ADDR_REG])
    )

  @property
  def last(self) -> AssemblyInstr | AssemblyLabel:
    return self.body[-1]

  def append(self, opname: str, *args: Any) -> None:
    self.body.append(AssemblyInstr(opname, list(args)))

  def append_label(self, name: str) -> None:
    self.body.append(AssemblyLabel(name))

  def lw(self, target_reg: str, source_addr: str, offset: int = 0) -> None:
    self.append('lw', target_reg, f'{offset}({source_addr})')

  def ret(self) -> None:
    self.lw(RETURN_ADDR_REG, 'sp')
    self.append('addi', 'sp', 'sp', STACKSIZE_PLACEHOLDER)
    self.append('jr', RETURN_ADDR_REG)

  def mv(self, target_reg: str, source_reg: str) -> None:
    if target_reg == source_reg:
      return

    self.append('mv', target_reg, source_reg)

  def __repr__(self) -> str:
    if not self.is_terminated:
      self.ret()

    indent = ' ' * 4
    base_indent = ' ' * 2

    body = list(map(lambda e: e.to_string(indent), self.body))
    init = [
      f'{indent}addi sp, sp, -{STACKSIZE_PLACEHOLDER}',
      f'{indent}sw {RETURN_ADDR_REG}, 0(sp)'
    ]

    body = f'\n'.join(init + body)
    return f'{base_indent}{self.name}:\n{body}'.replace(
      STACKSIZE_PLACEHOLDER,
      str(self.stacksize)
    )

# class CValue:
#   def __init__(self, typ: CType, asm: str) -> None:
#     self.typ = typ
#     self.asm = asm
