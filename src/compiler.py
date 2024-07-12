class Source:
  def __init__(self, source_path: str) -> None:
    self.path = source_path

    with open(source_path, 'r') as i:
      self.code = i.read()

class Location:
  def __init__(self, source: Source, line: int, column: int) -> None:
    self.source = source
    self.line = line
    self.column = column

  def __repr__(self) -> str:
    return f'{self.source.path} [line: {self.line + 1}, column: {self.column + 1}]'

class CompilationError(Exception):
  def __init__(self, msg: str, loc: Location) -> None:
    super().__init__()

    self.msg = msg
    self.loc = loc

  def __repr__(self) -> str:
    return f'{self.loc}: {self.msg}'

class CompilationUnit:
  def __init__(self, source_path: str, output_path: str) -> None:
    from node import Node

    self.source = Source(source_path)
    self.output_path = output_path

    self.tokens: list[Node] = []
    self.nodes: list[Node] = []
    self.assembly: str = ''

  def asm(self) -> str:
    from clexer import Lexer
    from cparser import Parser
    from cemitter import Emitter

    lexer = Lexer(self)
    self.tokens = lexer.lex()

    parser = Parser(self)
    self.nodes = parser.parse()

    emitter = Emitter(self)
    self.assembly = emitter.emit()
    # print(self.assembly)

    return self.assembly
