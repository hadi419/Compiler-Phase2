from typing import Any, Callable, cast
from compiler import CompilationError, CompilationUnit, Location
from node import Node
from clexer import ASSIGNMENT_OPERATORS

STATEMENTS_WITHOUT_SEMICOLON = [
  ';', 'if_stmt', 'while_stmt',
  'for_stmt', 'switch_stmt',
  'block'
]

GLOBALS_WITHOUT_SEMICOLON = [
  'fn'
]

class Parser:
  def __init__(self, unit: CompilationUnit) -> None:
    self.unit = unit
    self.index = 0

  @property
  def cur(self) -> Node:
    return self.unit.tokens[self.index]

  @property
  def bck(self) -> Node:
    return self.unit.tokens[self.index - 1]

  def has_token(self, offset=0) -> bool:
    return self.index + offset < len(self.unit.tokens)

  def advance(self, count=1) -> None:
    self.index += count

  def eat(self) -> Node:
    self.advance()
    return self.bck

  def match(self, *tags: str) -> bool:
    return self.cur.tag in tags

  def eat_match(self, *tags: str) -> bool:
    if not self.match(*tags):
      return False

    self.advance()
    return True

  def eat_expect(self, *tags: str) -> Node:
    if not self.eat_match(*tags):
      joined = '", "'.join(tags)
      raise CompilationError(f'Expected "{joined}", found "{self.cur.tag}"', self.cur.loc)

    return self.bck

  def parse_sequence(self, collector: Callable[[], Node], opener: str, sep: str, closer: str) -> list[Node]:
    sequence = []

    self.eat_expect(opener)
    while not self.eat_match(closer):
      if len(sequence) > 0:
        self.eat_expect(sep)

      sequence.append(collector())

    return sequence

  def parse_enum_body(self) -> list[Node]:
    def parse_enum_member() -> Node:
      name = self.eat_expect('ident')
      val = self.parse_expr() if self.eat_match('=') else None

      return Node(
        'enum_member',
        {'name': name, 'val': val},
        name.loc
      )

    enum_members = self.parse_sequence(
      parse_enum_member,
      '{', ',', '}'
    )

    self.minimize_enum_body(enum_members)

    return enum_members

  # in c multiple declaration of the same
  # enum member is not an error, its value
  # is the last assigned
  def minimize_enum_body(self, enum_members: list[Node]) -> None:
    names = [m.val['name'].val for m in enum_members]

    i = 0
    while i < len(enum_members):
      member = enum_members[i]

      if names.count(member.val['name'].val) > 1:
        enum_members.pop(i)
        names.pop(i)
        continue

      i += 1

  def parse_enum_type(self) -> Node:
    loc = self.eat().loc
    name = None if self.match('{', ';') else self.eat_expect('ident')
    body = None if not self.match('{') else self.parse_enum_body()

    return Node(
      'enum_type',
      {'name': name, 'body': body},
      loc
    )

  def match_type(self) -> tuple[bool, Node | None]:
    match self.cur.tag:
      case 'void' | 'int':
        return True, self.eat()

      case 'enum':
        return True, self.parse_enum_type()

      case _:
        return False, None

  def parse_type(self) -> Node:
    if not (matches := self.match_type())[0]:
      raise CompilationError(f'Expected a type, found "{self.cur.tag}"', self.cur.loc)

    return matches[1] # type: ignore

  def parse_params(self) -> list[Node]:
    def parse_param() -> Node:
      typ = self.parse_type()
      name = self.eat_expect('ident')

      return Node(
        'param', {'typ': typ, 'name': name}, self.cur.loc
      )

    return self.parse_sequence(
      parse_param,
      '(', ',', ')'
    )

  def parse_bin(self, ops: list[str], collector: Callable[[], Node]) -> Node:
    left = collector()

    while self.cur.tag in ops:
      op = self.eat()
      right = collector()
      left = Node(
        'bin',
        {'left': left, 'op': op, 'right': right},
        op.loc
      )

    return left

  def parse_args(self) -> list[Node]:
    return self.parse_sequence(
      self.parse_expr,
      '(',
      ',',
      ')'
    )

  def parse_value(self) -> Node:
    match self.cur.tag:
      case 'ident' | 'num':
        value = self.eat()

      case '(':
        # skipping '('
        self.advance()
        value = self.parse_expr()
        # closing previous '('
        self.eat_expect(')')

      # unaries
      case '!' | '~' | '+' | '-' | '++' | '--':
        op = self.eat()
        val = self.parse_value()

        value = Node('prefix', {'op': op, 'val': val}, op.loc)

      case _:
        raise CompilationError(f'Expected value, found "{self.cur.tag}"', self.cur.loc)

    while self.match('++', '--', '(', '['):
      match self.cur.tag:
        case '++' | '--':
          value = Node('suffix', {'op': self.eat(), 'val': value}, self.bck.loc)

        case '(':
          loc = self.cur.loc
          value = Node('call', {'val': value, 'args': self.parse_args()}, loc)

        case '[':
          loc = self.cur.loc
          value = Node('array_indexing', {'val': value, 'index': self.parse_surrounded_expr('[]')}, loc)

        case _:
          raise NotImplementedError(self.cur)

    return value

  # operator precedence took from https://en.cppreference.com/w/c/language/operator_precedence
  def parse_binaries(self) -> Node:
    parse_arithmetic = lambda: self.parse_bin(
      ['<<', '>>'], lambda: self.parse_bin(
        ['+', '-'], lambda: self.parse_bin(
          ['*', '/', '%'], self.parse_value
        )
      )
    )

    parse_logic1 = lambda: self.parse_bin(
      ['==', '!='], lambda: self.parse_bin(
        ['>', '>=', '<=', '<'], parse_arithmetic
      )
    )

    parse_logic2 = lambda: self.parse_bin(
      ['|'], lambda: self.parse_bin(
        ['^'], lambda: self.parse_bin(
          ['&'], parse_logic1
        )
      )
    )

    return self.parse_bin(
      ['||'], lambda: self.parse_bin(
        ['&&'], parse_logic2
      )
    )

  def parse_expr(self) -> Node:
    expr = self.parse_binaries()

    if self.eat_match(*ASSIGNMENT_OPERATORS):
      op = self.bck
      right = self.parse_expr()

      expr = Node(
        'assignment',
        {'left': expr, 'op': op, 'right': right},
        op.loc
      )

    return expr

  def parse_return_stmt(self, loc: Location) -> Node:
    val = None if self.match(';') else self.parse_expr()

    return Node('return_stmt', val, loc)

  def parse_surrounded_expr(self, par='()') -> Node:
    return self.parse_surrounded(self.parse_expr, par)

  def parse_surrounded(self, collector: Callable[[], Any], par='()') -> Any:
    self.eat_expect(par[0])
    parsed = collector()
    self.eat_expect(par[1])

    return parsed

  def parse_if_stmt(self, loc: Location) -> Node:
    cond = self.parse_surrounded_expr()
    body = self.parse_block()
    else_body = self.parse_block() if self.eat_match('else') else None

    return Node(
      'if_stmt',
      {'cond': cond, 'body': body, 'else_body': else_body},
      loc
    )

  def parse_for_stmt(self, loc: Location) -> Node:
    def parse_for_head() -> tuple[Node | None, Node | None, Node | None]:
      left = None
      cond = None
      right = None

      if (matches := self.match_type())[0]:
        left = self.parse_variable_stmt(cast(Node, matches[1]), self.eat_expect('ident'))
      elif not self.match(';'):
        left = self.parse_expr_stmt()

      self.eat_expect(';')

      if not self.match(';'):
        cond = self.parse_expr()

      self.eat_expect(';')

      if not self.match(')'):
        right = self.parse_expr_stmt()

      return left, cond, right

    left, cond, right = self.parse_surrounded(parse_for_head)
    body = self.parse_block()

    return Node(
      'for_stmt',
      {'left': left, 'cond': cond, 'right': right, 'body': body},
      loc
    )

  def parse_while_stmt(self, loc: Location) -> Node:
    cond = self.parse_surrounded_expr()
    body = self.parse_block()

    return Node(
      'while_stmt',
      {'cond': cond, 'body': body},
      loc
    )

  def parse_switch_case_or_default_body(self) -> list[Node]:
    stmts = []

    while not self.match('case', 'default', '}'):
      stmts.append(self.parse_stmt())

    return stmts

  def parse_switch_body(self) -> tuple[list[Node], Node | None]:
    self.eat_expect('{')

    cases = []
    default_case = None

    while self.eat_match('case'):
      loc = self.bck.loc

      val = self.parse_expr()
      self.eat_expect(':')
      body = self.parse_switch_case_or_default_body()

      cases.append(Node(
        'switch_case_node',
        {'val': val, 'body': body},
        loc
      ))

    if self.eat_match('default'):
      loc = self.bck.loc
      self.eat_expect(':')
      body = self.parse_switch_case_or_default_body()

      default_case = Node(
        'switch_default_node',
        body,
        loc
      )

    self.eat_expect('}')

    return cases, default_case

  def parse_switch_stmt(self, loc: Location) -> Node:
    val = self.parse_surrounded_expr()
    cases, default_case = self.parse_switch_body()

    return Node(
      'switch_stmt',
      {'val': val, 'cases': cases, 'default_case': default_case},
      loc
    )

  def parse_stmt(self) -> Node:
    cur = self.eat()

    match cur.tag:
      case 'return':
        stmt = self.parse_return_stmt(cur.loc)

      case 'if':
        stmt = self.parse_if_stmt(cur.loc)

      case 'while':
        stmt = self.parse_while_stmt(cur.loc)

      case 'for':
        stmt = self.parse_for_stmt(cur.loc)

      case 'break' | 'continue':
        stmt = cur

      case 'switch':
        stmt = self.parse_switch_stmt(cur.loc)

      case '{':
        self.advance(-1)
        stmt = Node('block', self.parse_block(), cur.loc)

      case _:
        self.advance(-1)

        if (matches := self.match_type())[0]:
          stmt = self.parse_typed_stmt(matches[1]) # type: ignore
        else:
          stmt = self.parse_expr_stmt()

    if stmt.tag not in STATEMENTS_WITHOUT_SEMICOLON:
      self.eat_expect(';')

    return stmt

  def parse_expr_stmt(self) -> Node:
    val = self.parse_expr()

    return Node(
      'expr_stmt', val, val.loc
    )

  def parse_block(self) -> list[Node]:
    statements = []

    if not self.match('{'):
      return [self.parse_stmt()]

    self.eat_expect('{')
    while not self.eat_match('}'):
      statements.append(self.parse_stmt())

    return statements

  def parse_function_node(self, rettyp: Node, name: Node) -> Node:
    params = self.parse_params()
    body = None if self.eat_match(';') else self.parse_block()

    return Node(
      'fn',
      {'rettyp': rettyp, 'name': name, 'params': params, 'body': body},
      name.loc
    )

  def parse_array_lengths(self) -> list[Node]:
    sizes = []

    while self.eat_match('['):
      sizes.append(self.parse_expr())
      self.eat_expect(']')

    return sizes

  def parse_array_initializer_or_parse_expr(self, array_level: int) -> Node:
    if array_level == 0:
      return self.parse_expr()

    loc = self.cur.loc
    elems = self.parse_sequence(
      lambda: self.parse_array_initializer_or_parse_expr(array_level - 1),
      '{', ',', '}'
    )

    return Node('array_initializer', elems, loc)

  def parse_variable_stmt(self, typ: Node, name: Node) -> Node:
    array_lengths = self.parse_array_lengths()
    val = self.parse_array_initializer_or_parse_expr(len(array_lengths)) if self.eat_match('=') else None

    return Node(
      'var',
      {'val': val, 'typ': typ, 'name': name, 'array_lengths': array_lengths},
      name.loc
    )

  def parse_typed_stmt(self, typ: Node) -> Node:
    if self.match(';'):
      return typ

    name = self.eat_expect('ident')

    if self.match('('):
      return self.parse_function_node(typ, name)

    return self.parse_variable_stmt(typ, name)

  def next_global_node(self) -> Node:
    node = self.parse_typed_stmt(self.parse_type())

    if node.tag not in GLOBALS_WITHOUT_SEMICOLON:
      self.eat_expect(';')

    return node

  def parse(self) -> list[Node]:
    nodes = []

    while self.has_token():
      nodes.append(self.next_global_node())

    return nodes
