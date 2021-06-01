import typing

import acme
import acme.datasets
import acme.adders.reverb as acme_reverb
import acme.agents.replay as acme_replay
import acme.specs
import acme.types
import reverb


# deprecated
# def make_reverb_episode_replay(
#     environment_spec: acme.specs.EnvironmentSpec,
#     extra_spec: acme.types.NestedSpec = (),
#     batch_size: int = 1,
#     max_sequence_length: int = 100,
#     min_replay_size: int = 1,
#     max_replay_size: int = 100_000,
#     # TODO: make use of the discount
#     discount: float = 1.0,
#     prefetch_size: int = 4,
#     replay_table_name: str = "reverb_episode_replay",
# ) -> acme.agents.replay.ReverbReplay:
#     """
#     Sample one episode at a time, limited usage due to batch_size can't be large.
#     Not complete now.
#     """
#     # Create a replay server to add data to. This uses no limiter behavior in
#     # order to allow the Agent interface to handle it.
#     replay_table = reverb.Table(
#         name=replay_table_name,
#         sampler=reverb.selectors.Uniform(),
#         remover=reverb.selectors.Fifo(),
#         max_size=max_replay_size,
#         rate_limiter=reverb.rate_limiters.MinSize(min_replay_size),
#         signature=acme_reverb.EpisodeAdder.signature(
#             environment_spec, extra_spec
#         ),
#     )
#     server = reverb.Server([replay_table], port=None)

#     # The adder is used to insert observations into replay.
#     address = f"localhost:{server.port}"
#     client = reverb.Client(address)
#     adder = acme_reverb.EpisodeAdder(
#         client,
#         max_sequence_length=max_sequence_length,
#         priority_fns={replay_table_name: None},  # do not use priority for this table
#     )

#     # The dataset provides an interface to sample from replay.
#     data_iterator = acme.datasets.make_reverb_dataset(
#         table=replay_table_name,
#         server_address=address,
#         batch_size=batch_size,
#         prefetch_size=prefetch_size,
#     ).as_numpy_iterator()
#     return acme.agents.replay.ReverbReplay(server, adder, data_iterator, client=client)
