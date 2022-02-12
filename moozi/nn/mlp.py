# TODO: re-implement

# class MLPNet(hk.Module):
#     """
#     NOTE: input tensors are assumed to have batch dimensions
#     """

#     def __init__(self, spec: NeuralNetworkSpec):
#         super().__init__()
#         self.spec = spec

#     def repr_net(self, flattened_frames):
#         net = hk.nets.MLP(
#             output_sizes=[*self.spec.repr_net_sizes, self.spec.dim_repr],
#             name="repr",
#             activation=jnp.tanh,
#             activate_final=True,
#         )
#         return net(flattened_frames)

#     def pred_net(self, hidden_state):
#         pred_trunk = hk.nets.MLP(
#             output_sizes=self.spec.pred_net_sizes,
#             name="pred_trunk",
#             activation=jnp.tanh,
#             activate_final=True,
#         )
#         v_branch = hk.Linear(output_size=1, name="pred_v")
#         p_branch = hk.Linear(output_size=self.spec.dim_action, name="pred_p")

#         pred_trunk_out = pred_trunk(hidden_state)
#         value = jnp.squeeze(v_branch(pred_trunk_out), axis=-1)
#         policy_logits = p_branch(pred_trunk_out)
#         return value, policy_logits

#     def dyna_net(self, hidden_state, action):
#         dyna_trunk = hk.nets.MLP(
#             output_sizes=self.spec.dyna_net_sizes,
#             name="dyna_trunk",
#             activation=jnp.tanh,
#             activate_final=True,
#         )
#         trans_branch = hk.nets.MLP(
#             output_sizes=[self.spec.dim_repr],
#             name="dyna_trans",
#             activation=jnp.tanh,
#             activate_final=True,
#         )
#         reward_branch = hk.nets.MLP(
#             output_sizes=[1],
#             name="dyna_reward",
#             activation=jnp.tanh,
#             activate_final=True,
#         )

#         action_one_hot = jax.nn.one_hot(action, num_classes=self.spec.dim_action)
#         chex.assert_equal_rank([hidden_state, action_one_hot])
#         state_action_repr = jnp.concatenate((hidden_state, action_one_hot), axis=-1)
#         dyna_trunk_out = dyna_trunk(state_action_repr)
#         next_hidden_states = trans_branch(dyna_trunk_out)
#         next_rewards = jnp.squeeze(reward_branch(dyna_trunk_out), axis=-1)
#         return next_hidden_states, next_rewards

#     def initial_inference(self, init_inf_features: RootInferenceFeatures):
#         chex.assert_rank(init_inf_features.stacked_frames, 3)
#         flattened_stacked_frames = hk.Flatten()(init_inf_features.stacked_frames)
#         hidden_state = self.repr_net(flattened_stacked_frames)
#         value, policy_logits = self.pred_net(hidden_state)
#         reward = jnp.zeros_like(value)
#         chex.assert_rank([value, reward, policy_logits, hidden_state], [1, 1, 2, 2])
#         return NNOutput(
#             value=value,
#             reward=reward,
#             policy_logits=policy_logits,
#             hidden_state=hidden_state,
#         )

#     def recurrent_inference(self, recurr_inf_features: TransitionInferenceFeatures):
#         # TODO: a batch-jit that infers K times?
#         next_hidden_state, reward = self.dyna_net(
#             recurr_inf_features.hidden_state, recurr_inf_features.action
#         )
#         value, policy_logits = self.pred_net(next_hidden_state)
#         chex.assert_rank(
#             [value, reward, policy_logits, next_hidden_state], [1, 1, 2, 2]
#         )
#         return NNOutput(
#             value=value,
#             reward=reward,
#             policy_logits=policy_logits,
#             hidden_state=next_hidden_state,
#         )