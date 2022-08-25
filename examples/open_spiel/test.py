# %%
import jax
from moozi import planner, training_worker
from moozi.core.utils import fetch_device_array
from moozi.gii import GII
from moozi.tournament import Tournament, Player
from moozi.parameter_optimizer import load_all_params_and_states
from moozi.driver import Driver, get_config

# %%
lookup = load_all_params_and_states("/home/zeyi/assets/arena")
config = get_config()
config.training_worker.planner.num_simulations = 256
config.training_worker.planner.kwargs.dirichlet_frac = 0.1
config.training_worker.planner.kwargs.temperature = 0.25
driver = Driver.setup(config)

# %%
for k, v in lookup.items():
    lookup[k] = fetch_device_array(lookup[k])

# %%
players = []
for i, key in enumerate(lookup):
    if key != "latest":
        player = Player(
            name=key,
            params=lookup[key][0],
            state=lookup[key][1],
            planner=planner,
            elo=1300,
        )
        players.append(player)
print(len(players))

# %%
gii = GII(
    env_name=config.env.name,
    stacker=driver.stacker,
    planner=planner,
    params=None,
    state=None,
    random_key=jax.random.PRNGKey(0),
    backend="cpu",
)
t = Tournament(gii=gii, players=players)

# %%
results = t.run_round_robin(num_matches=2)

# %%
df = t.dataframe
df['Name'] = df['Name'].astype(int)

# %%
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from IPython import display
sns.set(font_scale=1.0, font="Trebichet MS")
p = sns.lineplot(
    x='Name',
    y='ELO',
    data=df,
    linewidth=2,
    alpha=0.7,
)
p.set(xlabel='Training Steps', ylabel='ELO')
plt.savefig('elo.png')
p
# %%
t.dataframe
# %%
