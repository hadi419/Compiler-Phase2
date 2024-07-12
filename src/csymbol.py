from typing import Any
from node import Location

SYMBOL_TAGS = [
  'fn', 'local', 'global',
  'enum', 'enum_member'
]

class CSymbol:
  def __init__(self, tag: str, is_prototype: bool, val: Any, loc: Location) -> None:
    assert tag in SYMBOL_TAGS

    self.tag = tag
    self.is_prototype = is_prototype
    self.val = val
    self.loc = loc
