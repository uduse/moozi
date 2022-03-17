# %%
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pprint import pprint, pformat
import tree
from haiku import transform
import numpy as np
from moozi.nn.nn import *
from moozi.nn.resnet import ResNetArchitecture, ResNetSpec
from acme.jax.utils import add_batch_dim, squeeze_batch_dim

# %%
def make_model_2(architecture_cls: Type[NNArchitecture], spec: NNSpec):
    arch = functools.partial(architecture_cls, spec)

    def multi_transform_target():
        module = arch()

        def module_walk(root_feats, trans_feats):
            root_out = module.root_inference(root_feats, is_training=True)
            trans_out = module.trans_inference(trans_feats, is_training=True)
            return (trans_out, root_out)

        return module_walk, (module.root_inference, module.trans_inference)

    transformed = hk.multi_transform_with_state(multi_transform_target)

    def init_params_and_state(rng):
        batch_dim = (1,)
        root_feats = RootFeatures(
            obs=np.ones(batch_dim + spec.obs_rows),
            player=np.array([0]),
        )
        trans_feats = TransitionFeatures(
            hidden_state=np.ones(
                batch_dim + spec.obs_rows[:-1] + (spec.dim_repr,)
            ),
            action=np.array([0]),
        )

        return transformed.init(rng, root_feats, trans_feats)

    dummy_random_key = jax.random.PRNGKey(0)

    def root_inference(params, state, feats, is_training):
        return transformed.apply[0](params, state, dummy_random_key, feats, is_training)

    def trans_inference(params, state, feats, is_training):
        return transformed.apply[1](params, state, dummy_random_key, feats, is_training)

    def root_inference_unbatched(params, state, feats, is_training):
        out, state = root_inference(params, state, add_batch_dim(feats), is_training)
        return squeeze_batch_dim(out), state

    def trans_inference_unbatched(params, state, feats, is_training):
        out, state = trans_inference(params, state, add_batch_dim(feats), is_training)
        return squeeze_batch_dim(out), state

    return NNModel(
        spec,
        init_params_and_state,
        root_inference,
        trans_inference,
        root_inference_unbatched,
        trans_inference_unbatched,
    )


# %%
def print_state_info(state):
    names = [
        "~_repr_net/conv_block/batch_norm/~/mean",
        "~_dyna_net/conv_block/batch_norm/~/mean",
        "~_pred_net/conv_block/batch_norm/~/mean",
    ]

    for name in names:
        for item in filter(lambda x: name in x[0], state.items()):
            print(item[0], item[1]["counter"])
    print()


# %%
model = make_model_2(
    ResNetArchitecture,
    ResNetSpec(obs_rows=(3, 3, 1), dim_repr=4, dim_action=3),
)


# %%
key = jax.random.PRNGKey(0)
params, state = model.init_params_and_state(key)


for i in range(2):
    out, state = model.root_inference_unbatched(
        params,
        state,
        RootFeatures(obs=np.random.random((3, 3, 1)), player=np.array(0)),
        is_training=True,
    )
    print_state_info(state)

    for _ in range(2):
        out, state = model.trans_inference_unbatched(
            params,
            state,
            TransitionFeatures(hidden_state=out.hidden_state, action=np.array(0)),
            is_training=True,
        )
        print_state_info(state)
# %%
