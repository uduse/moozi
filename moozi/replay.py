import typing

import acme
import acme.datasets
import acme.adders.reverb as acme_reverb
import acme.agents.replay as acme_replay
import acme.specs
import acme.types
import reverb


# def make_replay(env_spec: acme.specs.EnvironmentSpec):
#     replay_table = reverb.Table(
#         name=replay_table_name,
#         sampler=sampler,
#         remover=reverb.selectors.Fifo(),
#         max_size=max_replay_size,
#         rate_limiter=reverb.rate_limiters.MinSize(min_replay_size),
#         signature=adders.NStepTransitionAdder.signature(environment_spec, extra_spec),
#     )
#     server = reverb.Server([replay_table], port=None)

#     # The adder is used to insert observations into replay.
#     address = f"localhost:{server.port}"
#     client = reverb.Client(address)
#     adder = adders.NStepTransitionAdder(client, n_step, discount, priority_fns)

#     # The dataset provides an interface to sample from replay.
#     data_iterator = datasets.make_reverb_dataset(
#         table=replay_table_name,
#         server_address=address,
#         batch_size=batch_size,
#         prefetch_size=prefetch_size,
#     ).as_numpy_iterator()
#     return ReverbReplay(server, adder, data_iterator, client=client)
