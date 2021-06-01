r"""
Type hints precedence:
    - chex
    - typing
    - moozi
    - python

Avoid using acme.types, it's too complicated
"""
import typing

Action = typing.NewType("Action", int)
