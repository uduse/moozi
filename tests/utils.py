import functools
# import trio_asyncio


# def with_trio_asyncio(fn):
#     @functools.wraps(fn)
#     def _wrapper(*args, **kwargs):
#         return trio_asyncio.run(functools.partial(fn, *args, **kwargs))

#     return _wrapper
