# %%
import pandas as pd
import seaborn as sns
import re
import guild.ipy as guild

# %%
runs = guild.runs()
runs = runs[
    [
        re.match("\d+ : \d+", r) is not None
        for r in runs["label"]
    ]
]

# %%
scalars = runs.scalars_detail()

# %%
df = scalars[
    scalars["tag"].str.match("(replay/num_env_frames_added|eval/episode_return)")
]
df["label"] = df["run"].map(lambda x: x["label"])
df["id"] = df["run"].map(lambda x: x["id"])

# %%
# based on steps
df2 = df.copy()
df2 = df2[["tag", "step", "val", "label"]]
df2 = df2[df2["tag"] == "eval/episode_return"]


# %%
# df2 = df2[df2['step']]

# %%
sns.set(rc={"figure.dpi": 300, "savefig.dpi": 300})
sns.set(font_scale=1.0, font="Trebichet MS")
p = sns.relplot(
    x="step",
    y="val",
    hue="label",
    data=df2,
    linewidth=3,
    alpha=0.7,
    kind='line'
    # figsize=(16, 9)
)
p.set(ylabel="Average Return", xlabel="Number of Training Steps")

# %%
# based on env frames
df3 = df.copy()
df3 = df3[["tag", "step", "val", "label", 'id']]
df3 = df3[df3["tag"] == "eval/episode_return"]
df3 = df3.drop(columns='tag')
df3
# df3['step'] = df3.step.map(lambda x: int(x / 25 * 259 + 0.5))

# %% 
df4 = df.copy()
df4 = df[df["tag"] == "replay/num_env_frames_added"]
df4 = df4.rename(columns={'val': 'num_frames'})
df4 = df4.drop(columns='run')
df4 = df4.groupby(['step', 'label', 'id'], as_index=False).sum().reset_index()
df4

# %%
df.apply(lambda x:  x['label'].split(':').strip(''), axis=1)

# %%
df5 = pd.merge(df3, df4, how='inner', on=['num_frames', 'label', 'id'])
df5

# %%

# %%
sns.relplot(
    x='num_frames',
    y='val',
    hue='label',
    data=df5.drop(columns=['step', 'index'])
)

# %%
import matplotlib.font_manager
from IPython.core.display import HTML


def make_html(fontname):
    return "<p>{font}: <span style='font-family:{font}; font-size: 24px;'>{font}</p>".format(
        font=fontname
    )


code = "\n".join(
    [
        make_html(font)
        for font in sorted(
            set([f.name for f in matplotlib.font_manager.fontManager.ttflist])
        )
    ]
)

HTML("<div style='column-count: 2;'>{}</div>".format(code))
# %%


# %%
