from typing import Any
from compiler import Location

indent = ''

class Node:
  def __init__(self, tag: str, val: Any, loc: Location) -> None:
    self.tag = tag
    self.val = val
    self.loc = loc

  @property
  def is_prototype(self) -> bool:
    return self.val['body'] is None

  def __repr__(self) -> str:
    if not isinstance(self.val, dict):
        return f'Node(tag: {repr(self.tag)}, val: {repr(self.val)})'

    global indent

    indent += '  '
    result = f'Node(\n{indent}tag: {repr(self.tag)},\n'

    for k, v in self.val.items():
      if v is None:
        continue

      result += f'{indent}{k}: {repr(v)},\n'

    indent = indent[:-2]
    return result + f'{indent})'