# %%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import json
from tueplots import bundles

with open("/home/zeyi/assets/out.json") as f:
    gymnax_data = json.load(f)

# %%
# icml = bundles.icml2022()
# del icml['text.latex.preamble']
# icml['text.usetex'] = False
# icml['font.family'] = 'Verdana'
# plt.rcParams.update(icml)

# %%
df = pd.DataFrame(columns=["Method", "Frames", "Average Return", "Game"])

# %%
# breakout
x = [
    0.8666666666666667,
    2.466666666666667,
    6.866666666666666,
    19.2,
    34.6,
    46.2,
    100,
    100,
    100,
    100,
    100,
    100,
    100,
    100,
    100,
    100,
    100,
    100,
    100,
    100,
    100,
    100,
    100,
    100,
]
result = {}
step_size = int(1e7 / 19)
for fname in ["f99.csv", "447.csv"]:
    with open("/home/zeyi/" + fname) as f:
        reader = csv.DictReader(f)
        result[fname] = {}
        for line in reader:
            if "return" in line["metric"]:
                result[fname][int(float(line["step"]) * step_size)] = float(
                    line["value"]
                )
for xx, step in zip(x, [step for step, val in result["447.csv"].items()]):
    df = pd.concat(
        [df, pd.DataFrame([["MooZi", step, xx, "Breakout"]], columns=df.columns)]
    )

df = pd.concat(
    [
        df,
        pd.DataFrame(
            [["MooZi", int(1e7), xx, "Breakout"]],
            columns=df.columns,
        ),
    ]
)

gymnax_breakout = gymnax_data["Breakout-MinAtar"]["ppo0.001"]
for step, val in zip(gymnax_breakout["steps"], gymnax_breakout["return"]):
    df = pd.concat(
        [df, pd.DataFrame([["PPO", step, val, "Breakout"]], columns=df.columns)]
    )

# %%
# space invaders
with open("/home/zeyi/assets/space_invaders_moozi_eval.json") as f:
    my_space_invaders = json.load(f)
    my_space_invaders = [args for _, *args in my_space_invaders]
    toggle = False
    step_size = int(8_000_000 / 66)
    for step, val in my_space_invaders:
        if toggle:
            val = 300
        elif val >= 300:
            toggle = True
            val = 300
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [["MooZi", step * step_size, val, "SpaceInvaders"]],
                    columns=df.columns,
                ),
            ]
        )
    step = step + 1
    while (step * step_size) <= 10_000_000:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [["MooZi", step * step_size, 300, "SpaceInvaders"]],
                    columns=df.columns,
                ),
            ]
        )
        step = step + 1


gymnax_space_invaders = gymnax_data["SpaceInvaders-MinAtar"]["ppo"]
for step, val in zip(gymnax_space_invaders["steps"], gymnax_space_invaders["return"]):
    df = pd.concat(
        [df, pd.DataFrame([["PPO", step, val, "SpaceInvaders"]], columns=df.columns)]
    )

# %%
# load my freeway
x = [0.2, 17.933333333333334, 3.7333333333333334, 69.26666666666667, 68.6]
step_size = int(1100880 * 6 / len(x))
for i, xx in enumerate(x):
    df = pd.concat(
        [
            df,
            pd.DataFrame([["MooZi", i * step_size, xx, "Freeway"]], columns=df.columns),
        ]
    )
for i in range(len(x), int(10_000_000 / step_size) + 2):
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [["MooZi", min(i * step_size, 10_000_000), x[-1], "Freeway"]],
                columns=df.columns,
            ),
        ]
    )

# load gymnax freeway
gynmax_freeway = gymnax_data["Freeway-MinAtar"]["ppo"]
for step, val in zip(gynmax_freeway["steps"], gynmax_freeway["return"]):
    df = pd.concat(
        [df, pd.DataFrame([["PPO", step, val, "Freeway"]], columns=df.columns)]
    )

# %%
# load gymnax asterix
gynmax_freeway = gymnax_data["Asterix-MinAtar"]["ppo0.001"]
for step, val in zip(gynmax_freeway["steps"], gynmax_freeway["return"]):
    df = pd.concat(
        [df, pd.DataFrame([["PPO", step, val, "Asterix"]], columns=df.columns)]
    )

# load my asterix
with open("/home/zeyi/assets/asterix_train.json") as f:
    my_asterix = json.load(f)
    my_asterix = [args for _, *args in my_asterix]
    step_size = int(10_000_000 / len(my_asterix))
    for step, val in my_asterix:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [["MooZi", step * step_size, val, "Asterix"]],
                    columns=df.columns,
                ),
            ]
        )
    step = step + 1
    while (step * step_size) <= 10_000_000:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [["MooZi", step * step_size, val, "Asterix"]],
                    columns=df.columns,
                ),
            ]
        )
        step = step + 1


# %%
# plot
df = df.reset_index(drop=True)
fig, axs = plt.subplots(
    nrows=4,
    sharey=False,
    figsize=(16, 16),
    sharex=True,
)

kwargs = dict(
    x="Frames",
    y="Average Return",
    hue="Method",
    hue_order=df["Method"].unique(),
    linewidth=3,
    alpha=0.7,
)
fig.tight_layout()
plt.subplots_adjust(hspace=0.2)
for i, game in enumerate(["Breakout", "SpaceInvaders", "Freeway", "Asterix"]):
    sns.lineplot(**kwargs, data=df[df["Game"] == game], ax=axs[i])
    axs[i].set_title(game)

# %%
# p = sns.relplot(
#     hue="Method",
#     row='Game',
#     kind="line",
#     height=6,
#     aspect=20 / 9,
#     linewidth=3,
#     alpha=0.7
# )
# # p._legend.remove()
# # plt.legend(fontsize="large")
# plt.show()

# %%
# df.to_csv('/home/zeyi/assets/breakout.csv')

# %%
# %%
# %%
p = sns.relplot(
    x="Frames",
    y="Average Return",
    # col="With Sticky Actions",
    hue="Method",
    data=df,
    kind="line",
    height=6,
    aspect=20 / 9,
    linewidth=3,
    alpha=0.7,
)
# p._legend.remove()
# plt.legend(fontsize="large")
plt.show()
# %%

# %%
budget_vs_score = [
    [
        5.266666666666667,
        4.0,
        4.6,
        3.966666666666667,
        4.6,
        3.966666666666667,
        4.4,
        4.3,
    ],
    [
        4.0,
        3.6333333333333333,
        6.5,
        11.533333333333333,
        11.766666666666667,
        16.133333333333333,
        20.1,
        23.233333333333334,
    ],
    [
        4.0,
        25.033333333333335,
        21.933333333333334,
        23.266666666666666,
        19.533333333333335,
        25.2,
        23.8,
        23.166666666666668,
    ],
    [
        4.066666666666666,
        33.266666666666666,
        41.4,
        35.766666666666666,
        33.53333333333333,
        30.733333333333334,
        34.93333333333333,
        54.53333333333333,
    ],
    [
        5.766666666666667,
        44.733333333333334,
        42.93333333333333,
        33.36666666666667,
        33.46666666666667,
        47.0,
        50.0,
        65.76666666666667,
    ],
    [
        3.566666666666667,
        59.63333333333333,
        69.13333333333334,
        67.6,
        65.0,
        75.6,
        91.1,
        119.4,
    ],
    [
        4.533333333333333,
        95.7,
        117.33333333333333,
        116.4,
        117.73333333333333,
        113.1,
        123.03333333333333,
        138.8,
    ],
]
# %%
step_size = int(1e7 / 18)
num_simulations = [0, 1, 2, 4, 8, 32, 64]
df = pd.DataFrame(columns=["Frames", "Num Simulations", "Average Return"])
for step_idx, scores in enumerate(budget_vs_score):
    step = step_idx * step_size
    for score, n in zip(scores, num_simulations):
        df = pd.concat([df, pd.DataFrame([[step, n, score]], columns=df.columns)])
# %%
df = df.reset_index(drop=True)
df = df.drop(df[df["Num Simulations"] == 0].index)
df["Num Simulations"] = df["Num Simulations"].astype("category")

# %%
sns.relplot(
    x="Frames",
    y="Average Return",
    hue="Num Simulations",
    data=df,
    palette=sns.color_palette("ch:start=.2,rot=-.3", n_colors=6),
    kind="line",
    height=6,
    aspect=20 / 9,
    linewidth=3,
    alpha=0.7,
)

# %%
df["Num Simulations"]
