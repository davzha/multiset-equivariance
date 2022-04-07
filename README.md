# Multiset-Equivariant Set Prediction with Approximate Implicit Differentiation

[[paper]][0] [[video]][1] [[poster]][9]

This repository contains the official implementation for [Multiset-Equivariant Set Prediction with Approximate Implicit Differentiation][0].
We propose a better permutation-equivariance property for multisets and improve an existing set predictor that has this property with approximate implicit differentiation.

## Requirements
Install the necessary python packages with either:
```
conda create --name multiset --file conda_requirements.txt
```
*or*
```
pip install -r pip_requirements.txt
```

Alternatively, install the appropriate [PyTorch][2] version for your system and run (note that this might install incompatible library versions):
```
pip install pytorch-lightning matplotlib scipy ray pandas wandb
```
## Reproducibility
### Pre-trained models
We make the [Weights & Biases][3] (wandb) tables containing all the training runs in the paper publicly available:
- Class-specific numbering: [Section 4.1 and Appendix E.1][4]
- Random sets autoencoding: [Section 4.2][5], [Appendix E.2][6]
- CLEVR object property prediction: [Section 4.3][7], [Appendix E.3][8]

You can look at the training curves for every run and download the corresponding model checkpoint to inspect the fully-trained weights.
The weights for some experiments maybe cannot be loaded by the corresponding training script due to some refactoring we did afterwards.

### Running the experiments yourself
**NOTE**: Make sure to adapt the `DATA_PATH` in `exp3_scripts/run_single.sh`. 
A free wandb account is required for some experiment evaluation scripts.

To reproduce the experiments including repeats over random seeds, run the following commands from the main directory:
```
exp1_scripts/run_all.sh  # Class-specific numbering, section 4.1
exp2_scripts/run_all.sh  # Random sets autencoding, section 4.2
exp3_scripts/run_all.sh  # CLEVR object property prediction, section 4.3
```
This runs the individual model scripts, which simply call `run_single.sh` with the appropriate parameters. `run_single.sh` then gives all the hyperparameters to the corresponding training script `train_exp{1..3}.py`. You can also launch individual scripts for a single seed. For example:

```
# Experiment 1, iDSPN model with seed 42 for all dataset sizes
exp1_scripts/idspn.sh 42

# Experiment 2, iDSPN model (default with momentum) with seed 42 for all combinations of set size and dimensionality
exp2_scripts/idspn_with_momentum.sh 42

# Experiment 3, iDSPN model with seed 42 on 128x128 images
exp3_scripts/img_size_128.sh 42 
```

If you want to run this on a cluster, it probably makes sense to rename `run_single.sh` to something else and create a new `run_single.sh` that queues up the renamed script as a job.

### Analysis of results
In the `analysis` directory, you can find the scripts used to produce the result figures and tables in the paper. The quantitative results for experiment 1 and 3 can be read directly from the wandb tables.

- `ran-results.py` creates the LaTeX table of the random set autoencoding results and the two figures based on fixing `n` or `d`. It automatically downloads the test results from wandb.
- `plot-ran.py` plots example random set autoencoding results. It downloads the checkpoints automatically from wandb.
- `plot-clevr.py` plots the example CLEVR outputs. It requires checkpoints from the clevr experiments to be named as such in the script, which can be obtained by evaluating a pre-trained model for varying iterations with `python train_exp3.py <other arguments...> --eval_checkpoint <path to checkpoint> --decoder_val_iters <n>`.
- `slot_attention_bar_heights.py` calculates the reference Slot Attention results for varying loss weights from the Slot Attention paper. These numbers for varying the loss weight on the 3d coordinates were lost in that paper, but fortunately the bar heights appear to be exactly proportional.

## BibTeX
```
@inproceedings{
    zhang2022multisetequivariant,
    title={Multiset-Equivariant Set Prediction with Approximate Implicit Differentiation},
    author={Zhang, Yan and Zhang, David W and Lacoste-Julien, Simon and Burghouts, Gertjan J and Snoek, Cees GM},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=5K7RRqZEjoS}
}
```


[0]: https://arxiv.org/abs/2111.12193
[1]: https://www.youtube.com/watch?v=2iGmXmjaQus
[2]: https://pytorch.org/get-started/locally/
[3]: https://wandb.ai/
[4]: https://wandb.ai/wdz/log_numbering?workspace=user-wdz
[5]: https://wandb.ai/cyanogenoid/multiset-equivariance-random-sets?workspace=user-cyanogenoid
[6]: https://wandb.ai/cyanogenoid/multiset-equivariance-random-sets-extra?workspace=user-cyanogenoid
[7]: https://wandb.ai/cyanogenoid/multiset-equivariance-clevr?workspace=user-cyanogenoid
[8]: https://wandb.ai/cyanogenoid/multiset-equivariance-clevr-extra?workspace=user-cyanogenoid
[9]: https://raw.githubusercontent.com/davzha/multiset-equivariance/main/multiset_equivariance_poster.png