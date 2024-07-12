from typing import Callable, NoReturn, cast
from compiler import CompilationUnit, CompilationError
from node import Node, Location
from csymbol import CSymbol
from functools import reduce
from cdata import *

import operator

class Emitter:
  def __init__(self, unit: CompilationUnit) -> None:
    self.unit = unit
    self.index = 0

    self.section_text: list[AssemblyFn] = []
    self.section_data: list[str] = []
    self.globls: list[str] = []
    self.regstack = 0
    self.labels_counter = 0

    self.global_symbols: dict[str, CSymbol] = {}
    # the first scope is empty
    # it's always used as template
    # it will be copied but never used
    # the 'int' is the sp index
    self.local_scopes: list[dict[str, CSymbol]] = [{}]
    self.cur_fn: tuple[CType, list[CType]]

    self.break_links: list[str] = []
    self.continue_links: list[str] = []

  @property
  def cur(self) -> Node:
    return self.unit.nodes[self.index]

  @property
  def local_symbols(self) -> dict[str, CSymbol]:
    return self.local_scopes[-1]

  @property
  def cur_fn_rettyp(self) -> CType:
    return self.cur_fn[0]

  @property
  def cur_fn_params_typ(self) -> list[CType]:
    return self.cur_fn[1]

  @property
  def cur_text(self) -> AssemblyFn:
    return self.section_text[-1]

  def next_reg(self) -> str:
    assert self.regstack in range(len(VALUE_REGISTERS))

    reg = VALUE_REGISTERS[self.regstack]
    self.regstack += 1

    return reg

  def pop_reg(self) -> str:
    assert self.regstack - 1 in range(len(VALUE_REGISTERS))

    self.regstack -= 1
    reg = VALUE_REGISTERS[self.regstack]

    return reg

  def peek_reg(self) -> str:
    assert self.regstack - 1 in range(len(VALUE_REGISTERS))

    return VALUE_REGISTERS[self.regstack - 1]

  def has_global_node(self, offset=0) -> bool:
    return self.index + offset < len(self.unit.nodes)

  def advance(self, count=1) -> None:
    self.index += count

  def declare_global(self, name: str, loc: Location, symbol: CSymbol) -> None:
    if name in self.global_symbols and not self.global_symbols[name].is_prototype:
      raise CompilationError(f'Global identifier "{name}" already declared', loc)

    self.globl(name)
    self.global_symbols[name] = symbol

  def declare_enum_members(self, body: list[Node]) -> None:
    values = []

    for member in body:
      val = member.val['val']

      if val is None:
        val = 0 if len(values) == 0 else values[-1] + 1
      else:
        val = self.expect_const_int_and(
          self.fold(val),
          member.loc
        )

      self.declare_global(member.val['name'].val, member.loc, CSymbol(
        'enum_member', False, val, member.loc
      ))

      values.append(val)

  def evaluate_enum(self, typ: Node) -> CType:
    name = typ.val['name']
    body = typ.val['body']
    # enum are basically aliases for int
    res = CType('int', 4)

    if body is None:
      return res

    self.declare_enum_members(body)

    if name is None:
      return res

    self.declare_global(
      name.val, name.loc, CSymbol(
        'enum', body is None, name.val, typ.loc
      )
    )

    return res

  def evaluate_typ(self, typ: Node, can_be_void: bool = False) -> CType:
    match typ.tag:
      case 'void':
        if not can_be_void:
          raise CompilationError('Type "void" not allowed here', typ.loc)

        return CType('void')

      case 'int':
        return CType('int', 4)

      case 'enum_type':
        return self.evaluate_enum(typ)

      case _:
        raise NotImplementedError(typ.tag)

  def evaluate_fn_prototype(self, fn: Node) -> tuple[CType, list[CType]]:
    rettyp = self.evaluate_typ(fn.val['rettyp'], can_be_void=True)
    params_typ = [
      self.evaluate_typ(param.val['typ']) for param in fn.val['params']
    ]

    return rettyp, params_typ

  def raw_alloc_on_stack(self, byte_size: int) -> int:
    sp = self.cur_text.stacksize
    self.cur_text.stacksize += byte_size

    return sp

  def alloc_on_stack(self, typ: CType, is_array: bool, array_length: int) -> int:
    byte_size = typ.calculate_size()

    if is_array:
      byte_size *= array_length

    return self.raw_alloc_on_stack(byte_size)

  def declare_local(
    self,
    name: str,
    typ: CType,
    is_array: bool,
    array_length: int,
    loc: Location
  ) -> int:
    if name in self.local_symbols:
      raise CompilationError(f'Local identifier "{name}" already declared', loc)

    sp = self.alloc_on_stack(typ, is_array, array_length)
    self.local_symbols[name] = CSymbol(
      'local',
      False,
      {
        'typ': typ,
        'sp': sp,
        'is_array': is_array,
        'array_length': array_length
      },
      loc
    )

    return sp

  def check_type_compatibility(self, left_typ: CType, right_typ: CType, loc: Location) -> None:
    if left_typ == right_typ:
      return

    raise CompilationError(f'Type incompatibility: "{left_typ}" not compatible with "{right_typ}"', loc)

  def get_concrete_type_for_bin(self, left_typ: CType, right_typ: CType) -> CType:
    if left_typ.tag in LITERAL_CTYPES:
      return left_typ

    return right_typ

  def evaluate_bin(self, node: Node) -> CType:
    left = node.val['left']
    op = node.val['op']
    right = node.val['right']
    is_logical_bin = op.tag in ['&&', '||']

    if is_logical_bin:
      left_typ, right_typ = self.evaluate_logical_bin(left, op, right)
    else:
      left_typ = self.evaluate_expr(left)
      right_typ = self.evaluate_expr(right)

    self.check_type_compatibility(left_typ, right_typ, op.loc)

    if not is_logical_bin:
      self.emit_asm_for_bin(op.tag)

    return self.get_concrete_type_for_bin(left_typ, right_typ)

  def emit_asm_for_complex_bin(self, l: str, op: str, r: str) -> None:
    output = self.next_reg()

    match op:
      case '==':
        self.text('sub', output, l, r)
        self.text('seqz', output, output)

      case '!=':
        self.text('sub', output, l, r)
        self.text('snez', output, output)

      case '<':
        self.text('slt', output, l, r)

      case '<=':
        self.text('sgt', output, l, r)
        self.text('xori', output, output, 1)

      case '>':
        self.text('sgt', output, l, r)

      case '>=':
        self.text('slt', output, l, r)
        self.text('xori', output, output, 1)

      case '<<':
        self.text('sll', output, l, r)

      case '>>':
        self.text('sra', output, l, r)

      case _:
        raise NotImplementedError(op)

  def evaluate_logical_bin(self, left: Node, op: Node, right: Node) -> tuple[CType, CType]:
    left_typ = self.evaluate_expr(left)
    self.condcheck(left_typ, left.loc)
    self.text('snez', self.peek_reg(), self.peek_reg())

    finally_label = self.next_label()
    opname = {'&&': 'beqz', '||': 'bnez'}[op.tag]

    self.text(opname, l := self.peek_reg(), finally_label)

    right_typ = self.evaluate_expr(right)
    self.condcheck(right_typ, right.loc)
    self.text('snez', l, self.pop_reg())

    self.label(finally_label)

    return left_typ, right_typ

  def emit_asm_for_bin(self, op: str) -> None:
    r = self.pop_reg()
    l = self.pop_reg()

    if op in COMPLEX_BIN_OPERATORS:
      self.emit_asm_for_complex_bin(l, op, r)
      return

    if op in BINARY_OPERATORS:
      output = self.next_reg()
      opname = BINARY_OPERATORS[op]

      self.text(opname, output, l, r)
      return

    raise NotImplementedError(op)

  def get(self, name: str, loc: Location) -> CSymbol:
    # this allows variable shadowing
    for local_scope in reversed(self.local_scopes):
      if name in local_scope:
        return local_scope[name]

    if name in self.global_symbols:
      return self.global_symbols[name]

    raise CompilationError(f'Identifier "{name}" not declared', loc)

  def emit_asm_to_save_reg_to_stack(self, reg: str) -> int:
    # here 'byte_size=4' means that we are saving the whole reg
    # this codegen component targets 32 bit riscv machines
    sp = self.raw_alloc_on_stack(byte_size=4)
    self.text('sw', reg, self.asm_local(sp))

    return sp

  def save_registers_to_stack(self) -> list[int]:
    stack_indexes = []

    while self.regstack != 0:
      stack_indexes.append(
        self.emit_asm_to_save_reg_to_stack(self.pop_reg())
      )

    return stack_indexes

  def emit_asm_for_args_pass(self, args: list[Node], params_typ: list[CType], call_loc: Location) -> list[int]:
    if len(args) != len(params_typ):
      raise CompilationError(f'Expected "{len(params_typ)}" args, found "{len(args)}"', call_loc)

    stack_indexes = self.save_registers_to_stack()

    for arg, param_typ in zip(args, params_typ):
      arg_typ = self.evaluate_expr(arg)
      self.typecheck(param_typ, arg_typ, arg.loc)

    return stack_indexes

  def restore_registers_from_stack(self, stack_indexes: list[int]) -> None:
    for sp in stack_indexes:
      self.cur_text.lw(self.next_reg(), 'sp', sp)

  def evaluate_call(self, node: Node) -> CType:
    val = node.val['val']
    args = node.val['args']

    if val.tag != 'ident':
      raise NotImplementedError('calling an expression is not implemented yet')

    symbol = self.get(val.val, val.loc)

    if symbol.tag != 'fn':
      raise CompilationError('Expression to call must be function or function pointer', val.loc)

    assembly_name = symbol.val['assembly_name']
    rettyp = symbol.val['rettyp']
    params_typ = symbol.val['params_typ']

    saved_registers_stack_indexes = self.emit_asm_for_args_pass(args, params_typ, node.loc)
    self.text('call', assembly_name)
    self.regstack = 1

    # restoring the state after the call
    self.restore_registers_from_stack(saved_registers_stack_indexes)

    return rettyp

  def evaluate_ident(self, node: Node) -> CType:
    symbol = self.get(node.val, node.loc)
    is_array = isinstance(symbol.val, dict) and symbol.val['is_array']

    match symbol.tag:
      case 'local':
        sp = symbol.val['sp']

        if not is_array:
          self.cur_text.lw(self.next_reg(), 'sp', sp)
        else:
          output = self.next_reg()
          self.text('mv', output, 'sp')
          self.text('addi', output, output, sp)

      case 'global':
        self.text('la', r := self.next_reg(), symbol.val['name'])

        if not is_array:
          self.cur_text.lw(r, r)

      case 'enum_member':
        return self.evaluate_num(Node('num', symbol.val, node.loc))

      case _:
        raise CompilationError(f'This is not a variable', node.loc)

    return symbol.val['typ']

  def evaluate_expr(self, node: Node, can_be_void: bool = False) -> CType:
    match node.tag:
      case 'ident':
        typ = self.evaluate_ident(node)

      case 'num':
        typ = self.evaluate_num(node)

      case 'bin':
        typ = self.evaluate_bin(node)

      case 'suffix':
        typ = self.evaluate_suffix(node)

      case 'prefix':
        typ = self.evaluate_prefix(node)

      case 'assignment':
        typ = cast(
          CType,
          self.evaluate_or_process_assignment(node, return_val=True)
        )

      case 'call':
        typ = self.evaluate_call(node)

      case 'array_indexing':
        typ = self.evaluate_array_indexing(node)

      case _:
        raise NotImplementedError(node.tag)

    if can_be_void or typ != CType('void'):
      return typ

    raise CompilationError(f'Expression with inconcrete type "{typ}"', node.loc)

  def evaluate_array_indexing(self, node: Node) -> CType:
    val = node.val['val']
    index = node.val['index']

    array_typ = self.evaluate_expr(val)
    ref_to_array = self.peek_reg()

    index_typ = self.evaluate_expr(index)
    self.condcheck(index_typ, index.loc)
    offset = self.pop_reg()

    elem = self.peek_reg()
    self.text('li', 't0', array_typ.calculate_size())
    self.text('mul', offset, offset, 't0')
    self.text('add', ref_to_array, ref_to_array, offset)
    self.cur_text.lw(elem, ref_to_array)

    return array_typ

  def evaluate_suffix(self, node: Node) -> CType:
    op = node.val['op']
    val = node.val['val']

    return self.evaluate_or_process_assignment(
      self.build_xxcrement_node(op, val),
      return_val=False,
      load_lvalue_before_assign=True
    )

  def build_xxcrement_node(self, op: Node, val: Node) -> Node:
    assert op.tag in ['++', '--']

    by = f'{op.tag[0]}='
    return Node(
      'assignment',
      {
        'left': val,
        'op': Node(by, by, op.loc),
        'right': Node('num', 1, op.loc)
      },
      op.loc
    )

  def evaluate_prefix(self, node: Node) -> CType:
    op = node.val['op']
    val = node.val['val']

    if op.tag in ['++', '--']:
      return cast(CType, self.evaluate_or_process_assignment(
        self.build_xxcrement_node(op, val),
        return_val=True
      ))

    val_typ = self.evaluate_expr(val)

    match op.tag:
      case '+':
        pass

      case '-':
        self.text('neg', self.peek_reg(), self.peek_reg())

      case '~':
        self.text('not', self.peek_reg(), self.peek_reg())

      case '!':
        self.text('seqz', self.peek_reg(), self.peek_reg())

      case _:
        raise NotImplementedError(op.tag)

    return val_typ

  def pop_load_instruction(self, loc: Location, load_lvalue_before_assign: bool = False) -> AssemblyInstr:
    if (
      self.cur_text.last.is_instr and
      self.cur_text.last.opname not in LOAD_INSTRUCTIONS # type: ignore
    ):
      raise CompilationError('This is not a valid "lvalue"', loc)

    if load_lvalue_before_assign:
      return self.cur_text.last # type: ignore

    # discarding the result of this load instruction
    # we are cutting away
    self.pop_reg()
    return self.cur_text.body.pop() # type: ignore

  def evaluate_or_process_assignment(
    self,
    node: Node,
    return_val: bool,
    load_lvalue_before_assign: bool = False
  ) -> CType:
    left = node.val['left']
    op = node.val['op']
    right = node.val['right']

    if op.tag != '=':
      right = self.build_bin_node_for_assignment_by(
        left,
        op,
        right
      )

    rvalue = self.evaluate_expr(right)

    # evaluating left value and right value
    # but left value is turned into its address
    lvalue = self.evaluate_expr(left)
    load_instruction = self.pop_load_instruction(left.loc, load_lvalue_before_assign)

    self.typecheck(lvalue, rvalue, op.loc)

    lvalue_output = self.pop_reg() if load_lvalue_before_assign else None
    rvalue_output = (self.peek_reg if return_val else self.pop_reg)()

    # emitting assignment instruction
    self.text(
      store_to_load_instruction(load_instruction.opname),
      rvalue_output,
      load_instruction.args[1]
    )

    if lvalue_output is not None:
      self.cur_text.mv(self.next_reg(), lvalue_output)

    return rvalue

  def build_bin_node_for_assignment_by(self, left: Node, op: Node, right: Node):
    by = op.tag[:-1]
    right = Node(
      'bin',
      {'left': left, 'op': Node(by, by, op.loc), 'right': right},
      op.loc
    )

    return right

  def evaluate_num(self, node: Node) -> CType:
    output = self.next_reg()
    self.text('li', output, node.val)

    return CType('lit', 'int')

  def process_expr_as_stmt(self, node: Node) -> None:
    node = node.val

    match node.tag:
      case 'assignment':
        self.evaluate_or_process_assignment(node, return_val=False)

      # todo
      # case 'suffix'
      # case 'prefix'
      # case 'call'

      case _:
        self.evaluate_expr(node, can_be_void=True)
        # discarding the value
        self.pop_reg()

  def typecheck(self, expected: CType, actual: CType, loc: Location) -> None:
    if expected == actual:
      return

    raise CompilationError(f'Type mismatch: expected "{expected}", found "{actual}"', loc)

  def text(self, opname: str, *args: Any) -> None:
    self.cur_text.append(opname, *args)

  def data(self, line: str) -> None:
    self.section_data.append(line)

  def globl(self, name: str) -> None:
    if name in self.globls:
      return

    self.globls.append(name)

  def process_return_stmt(self, stmt: Node) -> None:
    if stmt.val is not None:
      typ = self.evaluate_expr(stmt.val)
      self.typecheck(self.cur_fn_rettyp, typ, stmt.val.loc)

      self.cur_text.mv(RETURN_REG, self.pop_reg())
    else:
      self.typecheck(self.cur_fn_rettyp, CType('void'), stmt.loc)

    self.cur_text.ret()

  def asm_local(self, sp: int) -> str:
    return f'{sp}(sp)'

  def evaluate_array_initializer_or_expr(
    self,
    node: Node,
    base_typ: CType,
    array_length: int,
    sp: int
  ) -> CType:
    if node.tag != 'array_initializer':
      return self.evaluate_expr(node)

    self.process_local_array_intializer(
      node.val,
      base_typ,
      array_length,
      sp,
      node.loc
    )
    return base_typ

  def process_local_array_intializer(
    self,
    elems: list[Node],
    base_typ: CType,
    array_length: int,
    sp: int,
    loc: Location
  ) -> None:
    if len(elems) > array_length:
      raise CompilationError('Too many elements in array initializer', loc)

    for elem in elems:
      typ = self.evaluate_array_initializer_or_expr(elem, base_typ, array_length, sp)
      self.typecheck(base_typ, typ, elem.loc)

      if elem.tag != 'array_initializer':
        self.text('sw', self.pop_reg(), self.asm_local(sp))
        sp += base_typ.calculate_size()
      else:
        sp += base_typ.calculate_size() * len(elem.val)

  def process_var_stmt(self, stmt: Node) -> None:
    typ = stmt.val['typ']
    name = stmt.val['name']
    val = stmt.val['val']
    array_lengths = stmt.val['array_lengths']
    is_array = len(array_lengths) > 0

    # evaluating the expected type
    typ_loc, typ = typ.loc, self.evaluate_typ(typ)
    array_length = self.fold_array_lengths(array_lengths)

    if val is not None:
      val_typ = self.evaluate_array_initializer_or_expr(val, typ, array_length, self.cur_text.stacksize)
      self.typecheck(typ, val_typ, typ_loc)

    sp = self.declare_local(name.val, typ, is_array, array_length, name.loc)
    if val is not None and not is_array:
      self.text('sw', self.pop_reg(), self.asm_local(sp))

  def next_label(self) -> str:
    self.labels_counter += 1
    return f'L{self.labels_counter}'

  def condcheck(self, cond_typ: CType, loc: Location) -> None:
    if cond_typ == CType('lit', 'int'):
      return

    raise CompilationError('Condition values must be integers or pointers', loc)

  def process_if_stmt(self, stmt: Node) -> None:
    cond = stmt.val['cond']
    body = stmt.val['body']
    else_body = stmt.val['else_body']
    has_else = else_body is not None

    else_label = self.next_label()
    finally_label = self.next_label() if has_else else else_label

    # evaluating the condition
    cond_typ = self.evaluate_expr(cond)
    self.condcheck(cond_typ, cond.loc)

    # branching based on the result of the condition
    self.text('beqz', self.pop_reg(), else_label)

    # then branch
    self.process_scoped(
      lambda: self.process_block(body)
    )

    if has_else:
      self.text('j', finally_label)

    # else branch
    self.label(else_label)

    if has_else:
      self.process_scoped(
        lambda: self.process_block(else_body)
      )

      self.label(finally_label)

  def process_for_stmt(self, stmt: Node) -> None:
    left = stmt.val['left']
    cond = stmt.val['cond']
    right = stmt.val['right']
    body = stmt.val['body']

    if cond is None:
      cond = Node('num', 1, stmt.loc)

    head_label = self.next_label()
    finally_label = self.next_label()

    def setup() -> None:
      if left is not None:
        self.process_stmt(left)

      # emitting the head of the loop
      self.label(head_label)
      cond_typ = self.evaluate_expr(cond)
      self.condcheck(cond_typ, cond.loc)
      self.text('beqz', self.pop_reg(), finally_label)

    def breakup() -> None:
      if right is not None:
        self.process_expr_as_stmt(right)

      # emitting the loop update
      self.text('j', head_label)

    self.continue_links.append(head_label)
    self.break_links.append(finally_label)
    # emitting the body of the loop
    self.process_scoped(
      lambda: self.process_block(body),
      setup,
      breakup
    )
    self.continue_links.pop()
    self.break_links.pop()

    # emitting end of the loop
    self.label(finally_label)

  def process_while_stmt(self, stmt: Node) -> None:
    cond = stmt.val['cond']
    body = stmt.val['body']

    head_label = self.next_label()
    finally_label = self.next_label()

    # head branch
    self.label(head_label)

    # evaluating the condition
    cond_typ = self.evaluate_expr(cond)
    self.condcheck(cond_typ, cond.loc)

    # branching based on the result of the condition
    self.text('beqz', self.pop_reg(), finally_label)

    self.continue_links.append(head_label)
    self.break_links.append(finally_label)
    # then branch
    self.process_scoped(
      lambda: self.process_block(body)
    )
    self.continue_links.pop()
    self.break_links.pop()

    # going back to the loop head
    self.text('j', head_label)

    # else branch
    self.label(finally_label)

  def label(self, name: str) -> None:
    self.cur_text.append_label(name)

  def process_stmt(self, stmt: Node) -> None:
    match stmt.tag:
      case 'return_stmt':
        self.process_return_stmt(stmt)

      case 'expr_stmt':
        self.process_expr_as_stmt(stmt)

      case 'var':
        self.process_var_stmt(stmt)

      case 'if_stmt':
        self.process_if_stmt(stmt)

      case 'while_stmt':
        self.process_while_stmt(stmt)

      case 'for_stmt':
        self.process_for_stmt(stmt)

      case 'block':
        self.process_scoped(
          lambda: self.process_block(stmt.val)
        )

      case 'switch_stmt':
        self.process_switch_stmt(stmt)

      case 'break' | 'continue':
        self.process_break_continue_stmt(stmt)

      case _:
        raise NotImplementedError(stmt.tag)

  def process_break_continue_stmt(self, stmt: Node) -> None:
    try:
      link = self.break_links[-1]

      if stmt.tag == 'continue':
        link = self.continue_links[-1]

      self.text('j', link)
    except IndexError:
      raise CompilationError(f'Statement "{stmt.tag}" can only be used in loops/switch', stmt.loc)

  def process_switch_stmt(self, stmt: Node) -> None:
    val = stmt.val['val']
    cases = stmt.val['cases']
    default_case = stmt.val['default_case']
    has_default_case = default_case is not None

    # building all labels
    finally_label = self.next_label()
    case_head_labels = [self.next_label() for _ in cases]
    case_body_labels = [self.next_label() for _ in cases]
    default_label = self.next_label() if has_default_case else finally_label

    case_head_labels.append(default_label)
    case_body_labels.append(default_label)

    self.break_links.append(finally_label)

    val_typ = self.evaluate_expr(val)
    val_reg = self.peek_reg()

    for i, case in enumerate(cases):
      self.label(case_head_labels[i])

      case_typ = self.evaluate_expr(case.val['val'])
      self.check_type_compatibility(val_typ, case_typ, case.loc)
      case_reg = self.pop_reg()

      self.text('bne', val_reg, case_reg, case_head_labels[i + 1])

      self.label(case_body_labels[i])
      self.process_block(case.val['body'])
      self.text('j', case_body_labels[i + 1])

    # discarding the switch val
    self.pop_reg()

    if has_default_case:
      self.label(default_label)
      self.process_block(default_case.val)

    self.break_links.pop()
    self.label(finally_label)

  def process_block(self, block: list[Node]) -> None:
    for stmt in block:
      self.process_stmt(stmt)

  def push_local_scope(self) -> None:
    self.local_scopes.append({})

  def pop_local_scope(self) -> None:
    self.local_scopes.pop()

  def predeclare_params(
    self,
    locals_to_predeclare: list[tuple[CType, Node]],
    locals_initializer: Callable[[int, int], None]
  ) -> None:
    for i, (local_typ, local_name) in enumerate(locals_to_predeclare):
      sp = self.declare_local(local_name.val, local_typ, False, 0, local_name.loc)
      locals_initializer(i, sp)

  def process_scoped(
    self,
    processor: Callable[[], None],
    setupper: Callable[[], None] = lambda: None,
    breakupper: Callable[[], None] = lambda: None
  ) -> None:
    self.push_local_scope()

    setupper()
    processor()
    breakupper()

    self.pop_local_scope()

  def get_nth_reg(self, index: int) -> str:
    return VALUE_REGISTERS[index]

  def process_global_fn(self, fn: Node) -> None:
    rettyp, params_typ = self.evaluate_fn_prototype(fn)
    name: Node = fn.val['name']
    symbol = CSymbol(
      'fn',
      fn.is_prototype,
      {'assembly_name': name.val, 'rettyp': rettyp, 'params_typ': params_typ},
      fn.loc
    )

    self.declare_global(name.val, name.loc, symbol)

    if fn.is_prototype:
      return

    self.regstack = 0
    self.append_assembly_fn(name.val)

    self.cur_fn = (rettyp, params_typ)
    self.process_scoped(
      # body generation
      lambda: self.process_block(fn.val['body']),
      lambda: self.predeclare_params(
        # here all the params to pre declare within the block
        [(param_typ, param.val['name']) for param_typ, param in zip(params_typ, fn.val['params'])],
        # params initializer
        lambda i, param_sp: self.text('sw', self.get_nth_reg(i), self.asm_local(param_sp))
      )
    )

  def raise_complex_initializer(self, loc: Location) -> NoReturn:
    raise CompilationError('Global variable initializer is too complex', loc)

  def fold_bin_from_op(
    self,
    left: tuple[Any, CType],
    op: str,
    right: tuple[Any, CType],
    loc: Location
  ) -> tuple[Any, CType]:

    if self.check_type_compatibility(left[1], right[1], loc):
      pass

    l = left[0]
    r = right[0]

    val = {
      '+': lambda: l + r,
      '-': lambda: l - r,
      '*': lambda: l * r,
      '/': lambda: l / r,
      '%': lambda: l % r,
      '^': lambda: l ^ r,
      '&': lambda: l & r,
      '|': lambda: l | r,

      '==': lambda: l == r,
      '!=': lambda: l != r,
      '<': lambda: l < r,
      '<=': lambda: l <= r,
      '>': lambda: l > r,
      '>=': lambda: l >= r
    }[op]()

    return val, self.get_concrete_type_for_bin(left[1], right[1])

  def fold_prefix_from_op(self, op: str, val: tuple[Any, CType], loc: Location) -> tuple[Any, CType]:
    v = val[0]
    t = val[1]

    val = {
      '+': lambda: +v,
      '-': lambda: -v,
      '~': lambda: ~v,
      '!': lambda: not v
    }[op]()

    return val, t

  def fold(self, node: Node, base_typ: CType | None = None) -> tuple[Any, CType]:
    match node.tag:
      case 'num':
        return node.val, CType('lit', 'int')

      case 'bin':
        return self.fold_bin_from_op(
          self.fold(node.val['left']),
          node.val['op'].val,
          self.fold(node.val['right']),
          node.loc
        )

      case 'prefix':
        return self.fold_prefix_from_op(
          node.val['op'].val,
          self.fold(node.val['val']),
          node.loc
        )

      case 'array_initializer':
        return (
          self.fold_array_initializer(node, base_typ), # type: ignore
          base_typ # type: ignore
        )

      case 'ident':
        return self.fold_ident(node)

      case _:
        self.raise_complex_initializer(node.loc)

  def fold_ident(self, node: Node) -> tuple[Any, CType]:
    symbol = self.get(node.val, node.loc)

    if symbol.tag != 'enum_member':
      self.raise_complex_initializer(node.loc)

    return symbol.val, CType('int', 4)

  def fold_array_initializer(self, node: Node, base_typ: CType) -> list[Any]:
    elems = []

    for elem in node.val:
      elem_val, elem_typ = self.fold(elem, base_typ)
      if isinstance(elem_val, list):
        elems.extend(elem_val)
        continue

      elems.append(elem_val)
      self.typecheck(base_typ, elem_typ, elem.loc)

    return elems

  def expect_const_int_and(self, const: tuple[Any, CType], loc: Location) -> int:
    self.typecheck(CType('lit', 'int'), const[1], loc)
    return const[0]

  def process_global_var(self, node: Node) -> None:
    typ = node.val['typ']
    name = node.val['name']
    val = node.val['val']
    array_lengths = node.val['array_lengths']

    var_typ = self.evaluate_typ(typ)
    array_length = self.fold_array_lengths(array_lengths)

    if val is None:
      literal_val = 0
    else:
      literal_val, val_typ = self.fold(val, var_typ)
      self.typecheck(var_typ, val_typ, typ.loc)

    is_array = isinstance(literal_val, list)
    self.declare_global(name.val, name.loc, CSymbol(
      'global',
      False,
      {
        'typ': var_typ,
        'name': name.val,
        'is_array': is_array,
        'array_length': array_length
      },
      node.loc
    ))

    if not is_array:
      self.data(f'{name.val}: .word {literal_val}')
      return

    for i in range(array_length):
      if i < len(literal_val):
        val = literal_val[i]
      else:
        val = 0

      if i > 0:
        prefix = (len(name.val) + 1) * ' '
      else:
        prefix = f'{name.val}:'

      self.data(f'{prefix} .word {val}')

  def fold_array_lengths(self, array_lengths: list[Node]) -> int:
    arr = [self.expect_const_int_and(self.fold(l), l.loc) for l in array_lengths]

    return reduce(
      operator.mul,
      arr,
      1
    )

  def append_assembly_fn(self, name: str) -> None:
    self.section_text.append(AssemblyFn(name))

  def process_global_node(self) -> None:
    match self.cur.tag:
      case 'fn':
        self.process_global_fn(self.cur)

      case 'var':
        self.process_global_var(self.cur)

      case 'enum_type':
        self.evaluate_typ(self.cur)

      case _:
        raise NotImplementedError(self.cur.tag)

  def build_assembly(self) -> str:
    text = '\n\n'.join(map(repr, self.section_text))
    globls = '\n.globl '.join(self.globls)
    data = '\n  '.join(self.section_data)

    return f'''
.file "{self.unit.source.path}"

.globl {globls}

.text
{text}

.data
  {data}
'''

  def emit(self) -> str:
    while self.has_global_node():
      self.process_global_node()
      self.advance()

    return self.build_assembly()
