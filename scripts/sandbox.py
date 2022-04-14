# %%
from ray import ObjectRef
import ray


@ray.remote
def put_something():
    return ray.put(10)

@ray.remote
def return_something():
    return 10

@ray.remote
def print_putted_thing(thing):
    if isinstance(thing, ObjectRef):
        print(ray.get(thing))
    else:
        print(thing)


print_putted_thing.remote(put_something.remote())
print_putted_thing.remote(return_something.remote())

# %%
