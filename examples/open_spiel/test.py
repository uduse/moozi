# %%
import jax
from moozi.core.utils import fetch_device_array
from moozi.gii import GII
from moozi.tournament import Tournament, Player
from moozi.parameter_optimizer import load_all_params_and_states, load_params_and_state
from moozi.driver import ConfigFactory, Driver, get_config

# %%
config = get_config()
factory = ConfigFactory(config)

# %%
lookup = load_all_params_and_states(
    "/home/zeyi/miniconda3/envs/moozi/.guild/runs/e95902811f16497c80397281a535f341/checkpoints"
)
planner = factory.make_training_planner()
players = []
for i, key in enumerate(lookup):
    if key != "latest":
        player = Player(
            name=key,
            params=lookup[key][0],
            state=lookup[key][1],
            planner=planner,
            elo=0,
        )
        players.append(player)
print(len(players))

# %%
gii = GII(
    env=factory.make_env(),
    stacker=factory.make_history_stacker(),
    planner=factory.make_training_planner(),
    params=None,
    state=None,
    random_key=jax.random.PRNGKey(0),
    device="cpu",
)
t = Tournament(gii=gii, players=players)

# %%
results = t.run_round_robin(num_matches=8)

# %%
df = t.dataframe
df["Name"] = df["Name"].astype(int)

# %%
import seaborn as sns
import matplotlib.pyplot as plt
from tueplots import bundles
import matplotlib

# plt.rcParams.update(bundles.aistats2022())
# %matplotlib inline

from IPython import display

plt.rcParams.update(matplotlib.rcParamsDefault)
plt.rcParams.update({"figure.dpi": 300, "figure.figsize": (11.7, 6.27)})
sns.set(font_scale=2.5, font="Verdana")
p = sns.lineplot(
    x="Name",
    y="ELO",
    data=df,
    linewidth=4,
    alpha=0.7,
)
p.set(xlabel="Training Steps", ylabel="Elo Rating")
plt.savefig("elo.png")
df.to_csv("elo.csv")

# %%
# for r n results
# %%
