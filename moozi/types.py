r"""
Type hints precedence:
    - chex
    - typing
    - moozi
    - python

Avoid using acme.types, it's too complicated (really? need to reconsider)
Maybe we should put all custom types here.
"""
import typing

Action = typing.NewType("Action", int)
