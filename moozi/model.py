# from utils import NetworkOutput

import haiku as hk


class Model(hk.Module):
    def repr_net(self, image):
        net = hk.nets.MLP(output_sizes=[16, 16, dim_repr], name='repr')
        return net(image)

    def pred_net(self, hidden_state):
        v_net = hk.nets.MLP(output_sizes=[16, 16, 1], name='pred_v')
        p_net = hk.nets.MLP(output_sizes=[16, 16, dim_actions], name='pred_p')
        return v_net(hidden_state), p_net(hidden_state)

    def dyna_net(self, hidden_state, action):
        state_action_repr = jnp.concatenate((hidden_state, action), axis=-1)
        transition_net = hk.nets.MLP(
            output_sizes=[16, 16, dim_repr], name='dyna_trans')
        reward_net = hk.nets.MLP(
            output_sizes=[16, 16, dim_repr], name='dyna_reward')
        return transition_net(state_action_repr), reward_net(state_action_repr)

    def initial_inference(self, image):
        hidden_state = self.repr_net(image)
        reward = 0
        value, policy_logits = self.pred_net(hidden_state)
        return NetworkOutput(
            value=value,
            reward=reward,
            policy_logits=policy_logits,
            hidden_state=hidden_state
        )

    def recurrent_inference(self, hidden_state, action):
        hidden_state, reward = self.dyna_net(hidden_state, action)
        value, policy_logits = self.pred_net(hidden_state)
        return NetworkOutput(
            value=value,
            reward=reward,
            policy_logits=policy_logits,
            hidden_state=hidden_state
        )
