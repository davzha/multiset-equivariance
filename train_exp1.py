from pathlib import Path
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

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
import ap


def cmdline_args():
    parser = ArgumentParser()
    # experiment config
    parser.add_argument('--project', default=None)
    parser.add_argument('--name', default='default')
    # dataset config
    parser.add_argument('--loss', choices=['hungarian_l2', 'hungarian_ce', 'hungarian_nl'], default='hungarian_l2')
    parser.add_argument('--set_size', type=int, default=64)
    parser.add_argument('--set_dim', type=int, default=64)
    parser.add_argument('--input_dim', type=int, default=4)
    parser.add_argument('--dataset_size', type=int, default=64000)
    parser.add_argument('--n_obj_per_sample', type=int, default=4)
    parser.add_argument('--rand_perm', action='store_true', default=False)
    # training config
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--checkpoint_path', default='checkpoints')
    parser.add_argument('--num_data_workers', type=int, default=0)
    parser.add_argument('--num_ray_workers', type=int, default=0)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    # model config
    parser.add_argument('--model', default='idspn', choices=['idspn', 'dspn', 'deepsets', 'lstm', 'transformer_with_pe', 'transformer_no_pe','transformer_rnd_pe'])
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=64)
    # idspn config
    parser.add_argument('--decoder_lr', type=float, default=1.0)
    parser.add_argument('--decoder_iters', type=int, default=20)
    parser.add_argument('--decoder_momentum', type=float, default=0.9)
    parser.add_argument('--decoder_val_iters', type=int, default=None)
    parser.add_argument('--decoder_grad_clip', type=float)
    # wandb config
    parser.add_argument('--no_wandb', dest='use_wandb', action='store_false')
    # eval config
    parser.add_argument('--eval_checkpoint', default=None)

    args = parser.parse_args()

    if args.project is None:
        args.project = f'log_numbering'
    
    args.identifier = f'{args.input_dim}classes_{args.dataset_size}samples_{args.model}_{args.hidden_dim}dim' + ('_DA' if args.rand_perm else '')

    return args 


class SetPredictionModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        self.net = self.get_model(args.model)

        self.trainset = data.ClassSpecificNumbering(n_samples=args.dataset_size, set_size=args.set_size, set_dim=args.input_dim, n_obj_per_sample=args.n_obj_per_sample, rand_perm=args.rand_perm)
        self.valset = data.ClassSpecificNumbering(n_samples=6400, set_size=args.set_size, set_dim=args.input_dim, n_obj_per_sample=args.n_obj_per_sample)
        self.testset = data.ClassSpecificNumbering(n_samples=64000, set_size=args.set_size, set_dim=args.input_dim, n_obj_per_sample=args.n_obj_per_sample)

    def get_model(self, model_type):
        hp = self.hparams
        if 'dspn' in model_type:
            input_enc_kwargs = dict(d_in=hp.input_dim, d_hid=hp.hidden_dim, d_latent=hp.latent_dim, set_size=hp.set_size, pool='fs')
            inner_obj_kwargs = dict(d_in=hp.input_dim+hp.set_dim, d_hid=hp.hidden_dim, d_latent=hp.latent_dim, set_size=hp.set_size, 
                pool='fs', objective_type='mse_cat_input')
            dspn_kwargs = dict(learn_init_set=False, set_dim=hp.set_dim, set_size=hp.set_size, momentum=hp.decoder_momentum, lr=hp.decoder_lr, 
                iters=hp.decoder_iters, grad_clip=hp.decoder_grad_clip, projection='simplex', implicit=model_type=='idspn')
            net = models.DSPNBaseModel(input_enc_kwargs=input_enc_kwargs, inner_obj_kwargs=inner_obj_kwargs, dspn_kwargs=dspn_kwargs)
        elif model_type == 'deepsets':
            net = models.DSModel(hp.input_dim, hp.hidden_dim, hp.set_dim)
        elif model_type == 'lstm':
            net = models.LSTMModel(hp.input_dim, hp.hidden_dim, hp.set_dim)
        elif model_type == 'transformer_with_pe':
            net = models.TransformerWithPEModel(hp.input_dim, hp.hidden_dim, hp.set_dim, hp.set_size)
        elif model_type == 'transformer_no_pe':
            net = models.TransformerNoPEModel(hp.input_dim, hp.hidden_dim, hp.set_dim, hp.set_size)
        elif model_type == 'transformer_rnd_pe':
            net = models.TransformerRandomPEModel(hp.input_dim, hp.hidden_dim, hp.set_dim, hp.set_size)

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

        loss, indices = losses.hungarian_loss_numbering(
            batch[0], output, gt_output, 
            num_workers=self.args.num_ray_workers, 
            ret_indices=True,
            loss_type=self.args.loss.split('_')[-1])
        loss = loss.mean(0)
        micro_acc = losses.hungarian_micro_accuracy(output, gt_output, indices)
        macro_acc = losses.hungarian_macro_accuracy(output, gt_output, indices)

        log_dict = dict(loss=loss, micro_acc=micro_acc.mean(0), macro_acc=macro_acc.mean(0))
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
    
    def test_dataloader(self):
        return DataLoader(
            self.testset, 
            batch_size=self.args.batch_size, 
            shuffle=False,
            num_workers=self.args.num_data_workers,
        )


def train(args):
    model = SetPredictionModel(args)

    if args.num_ray_workers > 0:
        ray.init(num_cpus=args.num_ray_workers, include_dashboard=False)

    if args.use_wandb:
        run = wandb.init(
            name=args.name,
            project=args.project,
            reinit=False,
            # settings=wandb.Settings(start_method='fork'),
        )
        run.define_metric('macro_acc/val', summary='max')
        run.define_metric('micro_acc/val', summary='max')
        logger = WandbLogger(log_model=True)
        logger.watch(model.net)
        wandb.config.update(args)

    checkpoint_path = Path(args.checkpoint_path) / args.project / args.name
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gpus=1,
        num_nodes=1,
        logger=logger if args.use_wandb else None,
        callbacks=[
            ModelCheckpoint(dirpath=checkpoint_path, monitor='macro_acc/val', mode='max'),
        ],
        check_val_every_n_epoch=args.check_val_every_n_epoch,
    )

    trainer.fit(model)
    trainer.test()  # test best model
    return model


def test(args, model=None, trainer=None):
    if model is None:
        model = SetPredictionModel.load_from_checkpoint(checkpoint_path=args.eval_checkpoint, args=args)
    if trainer is None:
        trainer = pl.Trainer(gpus=1, num_nodes=1, logger=None)
    trainer.limit_test_batches = 1.0
    trainer.test(model)


def main():
    args = cmdline_args()
    pl.seed_everything(args.seed)

    if args.eval_checkpoint is None:
        train(args)
    else:
        test(args)


if __name__ == '__main__':
    main()