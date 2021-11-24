import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import idspn
from dspn import DSPN


def create_pairs(a, b=None):
    if b is None:
        b = a
    LA = a.size(1)
    LB = b.size(1)
    a = a.unsqueeze(2).expand(-1, -1, LB, -1)
    b = b.unsqueeze(1).expand(-1, LA, -1, -1)
    return a, b


class FSPool(nn.Module):
    """
        Simplified version of featurewise sort pooling, without the option of variable-size sets through masking. From:
        FSPool: Learning Set Representations with Featurewise Sort Pooling.
        Yan Zhang, Jonathon Hare, Adam Prügel-Bennett
        https://arxiv.org/abs/1906.02795
        https://github.com/Cyanogenoid/fspool
    """

    def __init__(self, in_channels, set_size):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(set_size, in_channels))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='linear')

    def forward(self, x):
        x, _ = x.sort(dim=1)
        x = torch.einsum('nlc, lc -> nc', x, self.weight)
        return x


class FSEncoder(nn.Module):
    def __init__(self, input_channels, dim, output_channels, set_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_channels, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, output_channels),
        )
        self.pool = FSPool(output_channels, set_size)

    def forward(self, x):
        x = self.mlp(x)
        x = self.pool(x)
        return x


class RNFSEncoder(FSEncoder):
    def __init__(self, input_channels, dim, output_channels, set_size):
        super().__init__(2 * input_channels, dim, output_channels, set_size ** 2)

    def forward(self, x):
        x = torch.cat(create_pairs(x), dim=-1).flatten(1, 2)
        x = super().forward(x)
        return x


class SumEncoder(nn.Module):
    def __init__(self, input_channels, dim, output_channels, set_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_channels, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, output_channels),
        )

    def forward(self, x):
        x = self.mlp(x)
        x = x.sum(1)
        return x


class MeanEncoder(nn.Module):
    def __init__(self, input_channels, dim, output_channels, set_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_channels, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, output_channels),
        )

    def forward(self, x):
        x = self.mlp(x)
        x = x.mean(1)
        return x

class ImageModel(nn.Module):
    """ ResNet18-based image encoder to turn an image into a feature vector """

    def __init__(self, latent, image_size):
        super().__init__()
        resnet = torchvision.models.resnet18()
        self.layers = nn.Sequential(*list(resnet.children())[:-2])
        resnet_output_dim = 512
        spatial_size = image_size // 32  # after resnet
        spatial_size = spatial_size // 2  # after strided conv
        self.end = nn.Sequential(
            nn.BatchNorm2d(resnet_output_dim),
            # now has 2x2 spatial size
            nn.Conv2d(resnet_output_dim, latent // spatial_size**2, 2, stride=2),
            # now has shape (n, latent // 4, 2, 2)
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.end(x)
        return x.view(x.size(0), -1)


class DSPNBaseModel(nn.Module):
    def __init__(self, input_enc_kwargs, inner_obj_kwargs, dspn_kwargs):
        super().__init__()
        self.input_enc_kwargs = input_enc_kwargs
        self.inner_obj_kwargs = inner_obj_kwargs
        self.dspn_kwargs = dspn_kwargs

        self.input_to_z = self.get_input_to_z(**input_enc_kwargs)
        self.inner_obj = self.get_inner_obj(**inner_obj_kwargs)
        self.dspn = self.get_dspn(inner_obj=self.inner_obj, **dspn_kwargs)

    def get_input_to_z(self, d_in, d_hid, d_latent, set_size, pool=None):
        enc_class = {
            'fs': FSEncoder,
            'rnfs': RNFSEncoder,
            'sum': SumEncoder,
            'mean': MeanEncoder,
        }[pool]
        enc = enc_class(d_in, d_hid, d_latent, set_size)
        return enc

    def get_inner_obj(self, d_in, d_hid, d_latent, set_size, pool=None, objective_type="mse"):
        enc_class = {
            'fs': FSEncoder,
            'rnfs': RNFSEncoder,
            'sum': SumEncoder,
            'mean': MeanEncoder,
        }[pool]
        enc = enc_class(d_in, d_hid, d_latent, set_size)
        if objective_type == "mse":
            obj_class = idspn.MSEObjective
        elif objective_type == "mse_regularized":
            obj_class = idspn.MSEObjectiveRegularized
        elif objective_type == "mse_cat_input":
            obj_class = idspn.MSEObjectiveCatInput
        else:
            raise ValueError(f"{objective_type} not implemented")
        obj = obj_class(enc)
        return obj
        
    def get_dspn(self, learn_init_set, inner_obj, set_dim, set_size, momentum, lr, iters, grad_clip, projection, implicit):
        if projection is not None:
            if projection == "simplex":
                projection = idspn.ProjectSimplex.apply
            else:
                raise ValueError(f"{projection} not implemented")

        if implicit:
            dspn = idspn.iDSPN(
                learn_init_set=learn_init_set,
                set_dim=set_dim,
                set_size=set_size,
                inner_obj=inner_obj,
                optim_f=lambda p: torch.optim.SGD(p, lr=lr, momentum=momentum, nesterov=momentum > 0),
                optim_iters=iters,
                grad_clip=grad_clip,
                projection=projection
            )
        else:
            dspn = DSPN(
                learn_init_set=learn_init_set,
                set_dim=set_dim,
                set_size=set_size,
                iters=iters,
                lr=lr,
                momentum=momentum,
                projection=projection,
            )
        return dspn

    def forward(self, x):
        z = self.input_to_z(x)
        set_0 = self.dspn.get_init_set(z)
        obj_fn = self.inner_obj
        if isinstance(obj_fn, idspn.MSEObjectiveCatInput):
            obj_fn = lambda z, set_t: self.inner_obj(z, set_t, x)
        if isinstance(obj_fn, idspn.MSEObjectiveRegularized):
            obj_fn = lambda z, set_t: self.inner_obj(z, set_t, set_0)

        y, grad = self.dspn(obj_fn, z, set_0)
        return y, grad


class DSPNImageModel(DSPNBaseModel):
    def get_input_to_z(self, d_latent, image_size):
        enc = ImageModel(d_latent, image_size)
        return enc


class DSLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin0 = nn.Linear(d_in, d_out)
        self.lin1 = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.lin0(x) + self.lin1(x.mean(1, keepdim=True))

class DSModel(nn.Module):
    def __init__(self, d_in, d_hid, d_out):
        super().__init__()
        self.ds = nn.Sequential(
            DSLayer(d_in, d_hid),
            nn.ReLU(),
            DSLayer(d_hid, d_out)
        )
    
    def forward(self, x):
        return self.ds(x)


class LSTMModel(nn.Module):
    def __init__(self, d_in, d_hid, d_out, bidirectional=True):
        super().__init__()
        self.proj_in = nn.Linear(d_in, d_hid)
        self.rnn = nn.LSTM(
            input_size=d_hid,
            hidden_size=d_hid,
            num_layers=2,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.proj_out = nn.Linear(2*d_hid if bidirectional else d_hid, d_out)
    
    def forward(self, x):
        x = self.proj_in(x)
        out, _ = self.rnn(x)
        out = self.proj_out(out)
        return out


class PositionalEncoding(nn.Module):
    """Implementation is based on https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x.transpose(0,1)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x).transpose(0,1)


class TransformerWithPEModel(nn.Module):
    def __init__(self, d_in, d_hid, d_out, set_size):
        super().__init__()
        self.transformer = nn.Transformer(
            d_hid, nhead=8, num_encoder_layers=2, num_decoder_layers=2,
            dim_feedforward=2*d_hid, dropout=0., batch_first=True)
        self.pe = PositionalEncoding(d_hid, dropout=0.)
        self.proj_src = nn.Linear(d_in, d_hid)
        self.proj_tgt = nn.Linear(d_in, d_hid)
        self.proj_out = nn.Linear(d_hid, d_out)

    def forward(self, x):
        src = self.proj_src(x)
        tgt = self.pe(self.proj_tgt(x))
        x = self.transformer(src, tgt)
        x = self.proj_out(x)
        return x

class TransformerNoPEModel(nn.Module):
    def __init__(self, d_in, d_hid, d_out, set_size):
        super().__init__()
        self.transformer = nn.Transformer(
            d_hid, nhead=8, num_encoder_layers=2, num_decoder_layers=2,
            dim_feedforward=2*d_hid, dropout=0., batch_first=True)
        self.proj_src = nn.Linear(d_in, d_hid)
        self.proj_tgt = nn.Linear(d_in, d_hid)
        self.proj_out = nn.Linear(d_hid, d_out)

    def forward(self, x):
        src = self.proj_src(x)
        tgt = self.proj_tgt(x)
        x = self.transformer(src, tgt)
        x = self.proj_out(x)
        return x

class TransformerRandomPEModel(nn.Module):
    def __init__(self, d_in, d_hid, d_out, set_size):
        super().__init__()
        self.transformer = nn.Transformer(
            d_hid, nhead=8, num_encoder_layers=2, num_decoder_layers=2,
            dim_feedforward=2*d_hid, dropout=0., batch_first=True)
        self.proj_src = nn.Linear(d_in, d_hid)
        self.proj_tgt = nn.Linear(d_in, d_hid//2)
        self.proj_out = nn.Linear(d_hid, d_out)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        src = self.proj_src(x)
        tgt = self.proj_tgt(x)
        tgt = torch.cat([tgt, self.alpha+torch.exp(self.beta)*torch.randn_like(tgt)], dim=-1)
        x = self.transformer(src, tgt)
        x = self.proj_out(x)
        return x
