import typing

Player = typing.NewType("Player", int)

# class Player(object):
#     def __init__(self, val):
#         self._val = val

#     def __eq__(self, other):
#         self._val = other._val

#     def __gt__(self, other):
#         self._val > other._val

#     def __hash__(self):
#         hash(self._val)

#     def __repr__(self):
#         return f"Player<{self._val}>"

#     @property
#     def val(self):
#         return self._val

# class Player(int):
#     def __repr__(self):
#         return f"Player<{int(self)}>"
