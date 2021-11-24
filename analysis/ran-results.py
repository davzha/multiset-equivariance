# %%
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import wandb
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
})

# %%
api = wandb.Api()
runs = api.runs('random_sets')


# %%
config = {
    'set_size': [],
    'set_dim': [],
    'model': [],
    'decoder_momentum': [],
    'decoder_iters': [],
    'seed': [],
}
val_loss = []

for run in runs:
    if 'hidden' in run.tags:
        print(f'skipping {run.name} because it\'s hidden')
        continue
    if run.summary.get('trainer/global_step') != 19999:
        print(f'skipping {run.name} because it hasn\'t finished')
        continue
    
    for k, v in config.items():
        v.append(run.config[k])
    val_loss.append(run.summary['loss/val'])

# %%
config['val_loss'] = val_loss
runs_df = pd.DataFrame(config)
print(runs_df)
# %%
df = runs_df.copy()

df = df[(df['seed'] < 5)]  # optional while runs are going
groups = df.groupby(['set_size', 'set_dim', 'model', 'decoder_momentum', 'decoder_iters'])
counts = groups['seed'].count()
# print(counts[counts != 5])
# assert (counts == 5).all(), 'number of seeds inconsistent'

# ^ commented out because a few DSPN runs had to be excluded

means = groups.mean()['val_loss'].reset_index()
stds = groups.std()['val_loss'].reset_index()

# %%
def select(data, set_size, iteration, model):
    return data[
        (data['set_size'] == set_size) &
        (data['decoder_iters'] == iteration) &
        (data['model'] == model[0]) &
        (data['decoder_momentum'] == model[1])
    ]

# each set_size gets its own tabular
print(r'\begin{tabular}{lcccccc}')
print(r'\toprule')
print('Model & Iterations & ' + ' & '.join(f"$d={d}$" for d in [2, 4, 8, 16, 32]) + r'\\')
print('\midrule')
for set_size in [2, 4, 8, 16, 32]:
    print(r'& \multicolumn{2}{c}{' + f'$n = {set_size}$' + r'} \\')
    # each iteration gets a block of rows
    for iteration in [10, 20, 40]:
        # each model gets its columns
        for model in [('idspn', 0.9, 'iDSPN+mom'), ('idspn', 0.0, 'iDSPN'), ('dspn', 0.9, 'Baseline')]:
            mean = select(means, set_size, iteration, model)
            mean = mean.sort_values('set_dim')
            std = select(stds, set_size, iteration, model)
            std = std.sort_values('set_dim')
            ms = zip(mean['val_loss'].values, std['val_loss'].values)
            losses = [f'{m:.1e}\\tiny$\\pm${s:.0e}' for m, s in ms]
            losses = [x.replace('e-0', 'e-') for x in losses]

            columns = [model[2], str(iteration)] + losses
            if model[0] == 'dspn' and iteration != 40:
                print(r'\vspace{2mm}%')
            print(' & '.join(columns) + r'\\')
    if set_size != 32:
        print('\midrule')
        
print(r'\bottomrule')
print(r'\end{tabular}')

# %%
colors = ['tab:blue', 'tab:green', 'tab:orange']


def adjust_lightness(color, amount=0.5):
    # https://stackoverflow.com/a/49601444
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])



# %%
def plot(hparam='set_dim', value=None):
    options = ['set_dim', 'set_size']
    options.remove(hparam)
    other_hparam = options[0]

    reference = means[
        (means[hparam] == value) &
        (means['model'] == 'dspn') &
        (means['decoder_momentum'] == 0.9) &
        (means['decoder_iters'] == 10)
    ]

    bar_width = 0.1
    for i, model in enumerate([('idspn', 0.9, 'iDSPN+mom'), ('idspn', 0.0, 'iDSPN'), ('dspn', 0.9, 'Baseline')]):
        mean = means[
            (means[hparam] == value) &
            (means['model'] == model[0]) &
            (means['decoder_momentum'] == model[1])
        ]
        std = stds[
            (stds[hparam] == value) &
            (stds['model'] == model[0]) &
            (stds['decoder_momentum'] == model[1])
        ]
        for j, iters in enumerate([10, 20, 40]):
            m = mean[mean['decoder_iters'] == iters]
            s = std[std['decoder_iters'] == iters]

            relative_loss = m['val_loss'].values / reference['val_loss'].values
            relative_error = s['val_loss'].values / reference['val_loss'].values
            bar_position = np.log2(m[other_hparam]) + (i + 3 * j - 4) * bar_width
            color = adjust_lightness(colors[i], 1 + (2 - j) / 4)

            plt.bar(
                bar_position,
                relative_loss,
                color=color,
                width=bar_width,
                yerr=relative_error,
                error_kw={'linewidth': 1, 'alpha': 0.8}
            )
    plt.gca().set_xticks([1, 2, 3, 4, 5])
    plt.gca().set_xticklabels([2, 4, 8, 16, 32])
    plt.axhline(1, color='k', linewidth=1, linestyle='--', alpha=0.3)
    plt.xlabel('$n$' if hparam == 'set_dim' else '$d$')
    if value in {2, 8, 32}:  # only need to show y (tick) labels when the subplot is on the left
        plt.ylabel('relative validation loss')
    else:
        plt.gca().tick_params(labelleft=False)
    plt.ylim(0, 1.5)
    plt.title('${} = {}$'.format('d' if hparam == 'set_dim' else 'n', value))


legend = [
    ('it=10 iDSPN+mom', adjust_lightness(colors[0], 1 + (2 - 0) / 4)),
    ('it=10 iDSPN', adjust_lightness(colors[1], 1 + (2 - 0) / 4)),
    ('it=10 Baseline', adjust_lightness(colors[2], 1 + (2 - 0) / 4)),
    ('it=20 iDSPN+mom', adjust_lightness(colors[0], 1 + (2 - 1) / 4)),
    ('it=20 iDSPN', adjust_lightness(colors[1], 1 + (2 - 1) / 4)),
    ('it=20 Baseline', adjust_lightness(colors[2], 1 + (2 - 1) / 4)),
    ('it=40 iDSPN+mom', adjust_lightness(colors[0], 1 + (2 - 2) / 4)),
    ('it=40 iDSPN', adjust_lightness(colors[1], 1 + (2 - 2) /4)),
    ('it=40 Baseline', adjust_lightness(colors[2], 1 + (2 - 2) / 4)),
]
legend = [mpatches.Patch(label=name, color=color) for name, color in legend]

for x in ['set_dim', 'set_size']:
    plt.figure(figsize=(8, 8))
    for i, val in enumerate([2, 4, 8, 16, 32]):
        plt.subplot(3, 2, i + 1)
        plot(x, val)
    plt.tight_layout()
    plt.legend(handles=legend, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig(f'{x}.pdf')

# %%
matplotlib.rc('font', size=14, family='Times New Roman')

def plot_one(n, d, equal_compute=False, legend=False, bar_width=0.3, val_loss=False):
    colors = ['tab:orange', 'tab:green', 'tab:blue']
    for i, model in enumerate([('dspn', 0.9, 'Baseline'), ('idspn', 0.0, 'iDSPN'), ('idspn', 0.9, 'iDSPN+mom')]):
        mean = means[
            (means['set_size'] == n) &
            (means['set_dim'] == d) &
            (means['model'] == model[0]) &
            (means['decoder_momentum'] == model[1])
        ]
        std = stds[
            (stds['set_size'] == n) &
            (stds['set_dim'] == d) &
            (stds['model'] == model[0]) &
            (stds['decoder_momentum'] == model[1])
        ]
        for j, iters in enumerate([10, 20, 40]):
            m = mean[mean['decoder_iters'] == iters]
            s = std[std['decoder_iters'] == iters]
            bar_position = j + (i - 1) * bar_width
            if equal_compute:
                if model[0] == 'idspn':
                    bar_position -= 1
                if model[0] == 'dspn' and iters == 40 or model[0] == 'idspn' and iters == 10:
                    continue
            plt.bar(
                bar_position,
                m['val_loss'],
                color=colors[i],
                width=bar_width,
                yerr=s['val_loss'],
                error_kw={'linewidth': 2, 'alpha': 0.8}
            )
    if legend:
        legend = [
            ('Baseline', colors[0]),
            ('iDSPN', colors[1]),
            ('iDSPN+mom', colors[2]),
        ]
        legend = [mpatches.Patch(label=name, color=color) for name, color in legend]
        plt.legend(handles=legend)
    plt.gca().set_xticks([0, 1, 2] if not equal_compute else [0, 1])
    plt.gca().set_xticklabels([10, 20, 40] if not equal_compute else [r'1$\times$', r'2$\times$'])
    plt.xlabel('iterations' if not equal_compute else 'compute')
    if val_loss:
        plt.ylabel('val loss')
    plt.title(f'$n={n}, d={d}$')

plt.figure(figsize=(10, 2))
plt.subplot(1, 4, 1)
plot_one(8, 8, equal_compute=False, val_loss=True)
plt.subplot(1, 4, 2)
plot_one(8, 8, equal_compute=True)
plt.subplot(1, 4, 3)
plot_one(16, 32, equal_compute=False)

plt.tight_layout()
plt.legend(handles=legend, bbox_to_anchor=(1.2, 1), loc='upper left', borderaxespad=0.)
plt.savefig('equivariance.pdf')
# %%
