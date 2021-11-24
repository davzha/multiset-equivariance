# %%
import os
import matplotlib.pyplot as plt
import matplotlib

import torch
import wandb

plt.rcParams.update({
    "text.usetex": True,
})

# %%
hparams = '--dataset random --epochs 20 --lr 1e-3 --batch_size 128 --project random_sets --input_encoder fs --decoder_encoder fs --decoder_starting_set --decoder_iters 20'

api = wandb.Api()
runs = api.runs('random_sets')
runs = [run for run in runs if run.config['seed'] == 7 and run.config['decoder_iters'] == 20]

def find_run(runs, **conditions):
    for run in runs:
        checks = [run.config[k] == v for k, v in conditions.items()]
        if all(checks):
            return run
    return None


# %%
data = {}
for n in [2, 4, 8, 16, 32]:
    for d in [2, 4, 8, 16, 32]:
        for model, momentum in [('idspn', 0.9), ('idspn', 0.0), ('dspn', 0.9)]:
            # select a run
            run = find_run(runs,
                set_size=n,
                set_dim=d,
                model=model,
                decoder_momentum=momentum,
            )
            # select and download checkpoint
            print('found', n, d, model, momentum, run)
            artifact = next(run.logged_artifacts())
            artifact_path = artifact.download()
            if os.path.exists("current.ckpt"):
                os.remove("current.ckpt")
            os.rename(f'{artifact_path}/model.ckpt', 'current.ckpt')
            # run checkpoint on data
            os.system(f"python train.py {hparams} --nn_gpus 0 --model {model} --decoder_momentum {momentum} --set_size {n} --set_dim {d} --eval_checkpoint current.ckpt --save_predictions predictions.pth --dataset_size 1280 --seed {(n + 1) * d}")  # seed generates unseen data that is consistent for a specific n x d combination
            preds = torch.load('predictions.pth')
            data.setdefault((n, d), {})[model, momentum] = preds


# %%
matplotlib.rc('font', size=14, family='Times New Roman')
plt.figure(figsize=(12, 14))
i = 1
for n in [2, 4, 8, 16, 32]:
    for d in [2, 4, 8, 16, 32]:
        plt.subplot(6, 5, i)
        i += 1
        preds = data[n, d]
        gt = preds[('idspn', 0.9)][0][1][0]
        idspnm = preds[('idspn', 0.9)][0][0][0]
        idspn = preds[('idspn', 0.0)][0][0][0]
        dspn = preds[('dspn', 0.9)][0][0][0]

        markers = ['o', '1', '3', '2']
        colors = ['black', 'tab:orange', 'tab:green', 'tab:blue']
        sizes = [64, 64, 64, 64]
        labels = ['Ground-truth', 'DSPN', 'iDSPN', 'iDSPN + momentum']
        for j, model in enumerate([gt, dspn, idspn, idspnm]):
            if j == 2:  # don't plot iDSPN because it's so similar to DSPN
                continue
            plt.scatter(model[:, 0], model[:, 1], marker=markers[j], s=sizes[j], color=colors[j], alpha=1, facecolors='none' if j == 0 else None, label=labels[j])
            plt.gca().set_aspect('equal')
        plt.ylim(-3, 3)
        plt.xlim(-3, 3)
        plt.title(f'$n={n}, d={d}$')
plt.tight_layout()
plt.legend(bbox_to_anchor=(-1, -0.3), borderaxespad=0, ncol=4)
plt.savefig('random_sets.pdf')
    