from typing import Any, Callable
from compiler import CompilationError, CompilationUnit, Location
from node import Node

KEYWORDS = [
  'void', 'int', 'return',
  'if', 'else', 'while', 'for',
  'enum', 'switch', 'case', 'default',
  'continue', 'break'
]

ASSIGNMENT_OPERATORS = [
  '=', '+=', '-=', '*=', '/=', '%=',
  '&=', '^=', '|='
  # these operators are 3-chars composed
  # 3-chars punctuation are not implemented yet
  # '<<=', '>>=',
]

BASIC_PUNCTUATION = [
  '==', '!=', '>', '>=', '<', '<=',
  '<<', '>>', '&&', '||', '++', '--',
  ',', ';', '(', ')', '[', ']', '{', '}',
  '+', '-', '*', '/', '%', '=', '!', '~',
  '&', '^', '|', ':'
]

PUNCTUATION = BASIC_PUNCTUATION + ASSIGNMENT_OPERATORS

class Lexer:
  def __init__(self, unit: CompilationUnit) -> None:
    self.unit = unit
    self.index = 0
    self.line = 0
    self.index_of_first_char_of_line = 0

  @property
  def nxt(self) -> str:
    return self.unit.source.code[self.index + 1]

  @property
  def cur(self) -> str:
    return self.unit.source.code[self.index]

  @property
  def col(self) -> int:
    return self.index - self.index_of_first_char_of_line

  @property
  def loc(self) -> Location:
    return Location(self.unit.source, self.line, self.col)

  def advance(self, count=1) -> None:
    self.index += count

  def eat_whitespace(self) -> None:
    while self.has_char() and self.cur.isspace():
      c = self.cur
      self.advance()

      if c != '\n':
        continue

      self.line += 1
      self.index_of_first_char_of_line = self.index

  def is_first_ident_char(self, c: str) -> bool:
    return c.isalpha() or c == '_'

  def is_ident_char(self, c: str) -> bool:
    return self.is_first_ident_char(c) or c.isdigit()

  def collect_sequence_into(
      self,
      predicator: Callable[[], bool],
      token: Node,
      transform: Callable[[str], Any]=str
    ) -> None:
    token.val = ''

    while self.has_char() and predicator():
      token.val += self.cur
      self.advance()

    token.val = transform(token.val)
    # going back to the last character of the identifier
    # because it will be skipped by 'NextToken' at the end of the scope
    self.advance(-1)

  def try_make_keyword(self, token: Node):
    for kw in KEYWORDS:
      if token.val == kw:
        token.tag = kw
        break

  def collect_ident_into(self, token: Node) -> None:
    token.tag = 'ident'
    self.collect_sequence_into(lambda: self.is_ident_char(self.cur), token)

    # maybe the identifier is a keyword
    self.try_make_keyword(token)

  def search_punctuation_for(self, token: Node, actual: str) -> bool:
    for p in PUNCTUATION:
      if actual == p:
        token.tag = actual
        token.val = actual
        return True

    return False

  def collect_punctuation_into(self, token: Node) -> None:
    has_next = self.has_char(1)

    if has_next and self.search_punctuation_for(token, double := f'{self.cur}{self.nxt}'):
      self.advance()
    elif not self.search_punctuation_for(token, self.cur):
      raise CompilationError('Found bad character', token.loc)

  def collect_num_into(self, token: Node) -> None:
    token.tag = 'num'

    try:
      self.collect_sequence_into(
        lambda: self.is_ident_char(self.cur),
        token,
        transform=lambda s: int(s, base=0)
      )
    except ValueError:
      raise CompilationError('Malformed literal int', token.loc)

  def next_token(self) -> Node:
    self.eat_whitespace()

    token = Node('eof', None, self.loc)

    if not self.has_char():
      return token

    if self.is_first_ident_char(self.cur):
      self.collect_ident_into(token)
    elif self.cur.isdigit():
      self.collect_num_into(token)
    else:
      self.collect_punctuation_into(token)

    self.advance()
    return token

  def has_char(self, offset=0) -> bool:
    return self.index + offset < len(self.unit.source.code)

  def lex(self) -> list[Node]:
    tokens = []

    self.eat_whitespace()

    while self.has_char():
      token = self.next_token()

      if token.tag == 'eof':
        break

      tokens.append(token)

    return tokens
