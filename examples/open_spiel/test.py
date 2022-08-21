# %%
import jax
from moozi.gii import GII
from moozi.tournament import Tournament, Player
from moozi.parameter_optimizer import load_params_and_states
from lib import get_config
from moozi.driver import Driver

# %%
lookup = load_params_and_states()
config = get_config()
driver = Driver.setup(config)

# %%
for k, v in lookup.items():
    lookup[k] = jax.tree_util.tree_map(
        lambda x: jax.device_put(x, jax.devices("cpu")[0]), lookup[k]
    )

# %%
players = []
for i, key in enumerate(lookup):
    if i % 5 == 0 and key != "latest":
        player = Player(
            name=key,
            params=lookup[key][0],
            state=lookup[key][1],
            planner=driver.trp,
            elo=1300,
        )
        players.append(player)

# %%
gii = GII(
    env_name=config.env.name,
    stacker=driver.stacker,
    planner=driver.trp,
    params=None,
    state=None,
    random_key=jax.random.PRNGKey(0),
    backend="cpu",
)
t = Tournament(gii=gii, players=players)

# %%
results = t.run_round_robin(num_matches=1)

# %%
import seaborn as sns
import matplotlib.pyplot as plt
sns.lineplot(
    x='index',
    y='ELO',
    data=t.dataframe.reset_index()
)
plt.savefig('elo.png')

# %%
