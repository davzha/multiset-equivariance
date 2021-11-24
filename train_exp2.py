from pathlib import Path
import math
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

try:
    import wandb
except ImportError:
    print('wandb not available')
try:
    import ray
except ImportError:
    print('ray not available')

import data
import losses
import models


def cmdline_args():
    parser = ArgumentParser()
    # experiment config
    parser.add_argument('--project', default=None)
    parser.add_argument('--name', default='default')
    # dataset config
    parser.add_argument('--loss', choices=['hungarian', 'chamfer'], default='hungarian')
    parser.add_argument('--set_size', type=int, default=10)
    parser.add_argument('--set_dim', type=int, default=2)
    parser.add_argument('--dataset_size', type=int, default=64000)
    # training config
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--n_gpus', type=int, default=1)
    parser.add_argument('--lr_drop_epoch', type=int, default=None)
    parser.add_argument('--checkpoint_path', default='checkpoints')
    parser.add_argument('--num_data_workers', type=int, default=0)
    parser.add_argument('--num_ray_workers', type=int, default=0)
    # model config
    parser.add_argument('--model', default='idspn', choices=['idspn', 'dspn'])
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=512)
    # idspn config
    parser.add_argument('--decoder_lr', type=float, default=1.0)
    parser.add_argument('--decoder_iters', type=int, default=20)
    parser.add_argument('--decoder_momentum', type=float, default=0.9)
    parser.add_argument('--decoder_val_iters', type=int, default=None)
    parser.add_argument('--decoder_grad_clip', type=float)
    parser.add_argument('--decoder_learn_init_set', action='store_true')
    parser.add_argument('--decoder_pool', choices=['fs', 'rnfs', 'sum', 'mean'], default='fs')
    # wandb config
    parser.add_argument('--no_wandb', dest='use_wandb', action='store_false')
    # eval config
    parser.add_argument('--progress_num_examples', type=int, default=0)
    parser.add_argument('--progress_path', default='progress')
    parser.add_argument('--eval_checkpoint', default=None)
    parser.add_argument('--test_after_training', action='store_true')
    parser.add_argument('--save_predictions', type=str)

    args = parser.parse_args()
    
    assert args.set_size > 0
    assert args.set_dim > 0

    if args.project is None:
        args.project = f'random-dim{args.set_dim}-size{args.set_size}'
    return args


class SetPredictionModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.net = self.get_model(args.model)
        self.trainset = data.RandomMultisets(size=args.dataset_size, cardinality=args.set_size, dim=args.set_dim)
        self.valset = data.RandomMultisets(size=args.dataset_size // 10, cardinality=args.set_size, dim=args.set_dim)

    def get_model(self, model_type):
        hp = self.hparams
        input_enc_kwargs = dict(d_in=hp.set_dim, d_hid=hp.hidden_dim, d_latent=hp.latent_dim, set_size=hp.set_size, pool='fs')
        inner_obj_kwargs = dict(d_in=hp.set_dim, d_hid=hp.hidden_dim, d_latent=hp.latent_dim, set_size=hp.set_size, pool=hp.decoder_pool, 
            objective_type='mse_regularized' if hp.decoder_learn_init_set else 'mse')
        dspn_kwargs = dict(learn_init_set=hp.decoder_learn_init_set, set_dim=hp.set_dim, set_size=hp.set_size, momentum=hp.decoder_momentum, lr=hp.decoder_lr, 
            iters=hp.decoder_iters, grad_clip=hp.decoder_grad_clip, projection=None, implicit=model_type=='idspn')

        net = models.DSPNBaseModel(input_enc_kwargs=input_enc_kwargs, inner_obj_kwargs=inner_obj_kwargs, dspn_kwargs=dspn_kwargs)
        return net

    def forward(self, x):
        input, gt_output = x
        output = self.net(input)
        if isinstance(output, tuple):
            output, set_grad = output
        else:
            set_grad = None
        return output, gt_output, set_grad

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, '/train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, '/val')
    
    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, '/test')

    def step(self, batch, batch_idx, suffix):
        output, gt_output, set_grad = self(batch)

        if self.args.loss == 'hungarian':
            loss = losses.hungarian_loss(output, gt_output, num_workers=self.args.num_ray_workers).mean(0)
        else:
            loss = losses.chamfer_loss(output, gt_output).mean(0)
            
        if batch_idx == 0 and self.args.progress_num_examples > 0 and '/train' != suffix:
            path = Path(self.args.progress_path) / self.args.project / self.args.name / f"{self.global_step}.png"
            self.plot_pointset(output, gt_output, Path(path), n_examples=self.args.progress_num_examples)

        log_dict = dict(loss=loss)
        if set_grad is not None:
            log_dict['grad_norm'] = set_grad.norm(dim=[1, 2]).mean()
        self.log_dict({k+suffix: v for k,v in log_dict.items()})

        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return opt

    def train_dataloader(self):
        return DataLoader(
            self.trainset, 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            num_workers=self.args.num_data_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset, 
            batch_size=self.args.batch_size, 
            shuffle=False,
            num_workers=self.args.num_data_workers,
        )

    def plot_pointset(self, pred, target, filename, n_examples):
        n_rows = n_cols = math.ceil(n_examples ** 0.5)
        fig, axs = plt.subplots(n_rows, n_cols, squeeze=False, figsize=(15,15))
        
        pred = pred.cpu().transpose(1, 2)
        target = target.cpu().transpose(1, 2)

        lim = -3, 3

        for i in range(n_examples):
            ax = axs[i // n_cols, i % n_cols]
            ax.scatter(target[i, 0], target[i, 1], marker='o', s=5**2)
            ax.scatter(pred[i, 0], pred[i, 1], marker='x', s=5**2)
            ax.axis("equal")
            ax.set_xlim(*lim)
            ax.set_ylim(*lim)
        
        fig.tight_layout()
        filename.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename)


def train(args):
    model = SetPredictionModel(args)

    if args.num_ray_workers > 0:
        ray.init(num_cpus=args.num_ray_workers, include_dashboard=False)

    if args.use_wandb:
        wandb.init(
            name=args.name,
            project=args.project,
            reinit=False,
            # settings=wandb.Settings(start_method="fork"),
        )
        logger = WandbLogger(log_model=True)
        logger.watch(model.net)
        wandb.config.update(args)

    checkpoint_path = Path(args.checkpoint_path) / args.project / args.name
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gpus=args.n_gpus,
        num_nodes=1,
        logger=logger if args.use_wandb else None,
        callbacks=[
            ModelCheckpoint(dirpath=checkpoint_path),
        ],
    )

    trainer.fit(model)
    if args.test_after_training:
        test(args, model, trainer)
    return model


def test(args, model=None, trainer=None):
    if model is None:
        model = SetPredictionModel.load_from_checkpoint(checkpoint_path=args.eval_checkpoint, args=args)
    if trainer is None:
        trainer = pl.Trainer(gpus=args.n_gpus, num_nodes=1)
    trainer.limit_val_batches = 1.0
    if not args.save_predictions:
        trainer.test(model, model.val_dataloader())
    else:
        outputs = trainer.predict(model, model.val_dataloader())
        torch.save([[o.cpu().detach() for o in output] for output in outputs], args.save_predictions)


def main():
    args = cmdline_args()
    pl.seed_everything(args.seed)

    if args.eval_checkpoint is None:
        train(args)
    else:
        test(args)


if __name__ == "__main__":
    main()
