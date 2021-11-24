import torch
import torch.nn as nn
import torch.nn.functional as F
import higher


class InnerSet(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def forward(self):
        return self.mask


class DSPN(nn.Module):
    """ Deep Set Prediction Networks
    Yan Zhang, Jonathon Hare, Adam Prügel-Bennett
    NeurIPS 2019
    https://arxiv.org/abs/1906.06565
    """

    def __init__(self, set_dim, set_size, learn_init_set, iters, lr, momentum, projection):
        """
        encoder: Set encoder module that takes a set as input and returns a representation thereof.
            It should have a forward function that takes two arguments:
            - a set: FloatTensor of size (batch_size, input_channels, maximum_set_size). Each set
            should be padded to the same maximum size with 0s, even across batches.
            - a mask: FloatTensor of size (batch_size, maximum_set_size). This should take the value 1
            if the corresponding element is present and 0 if not.

        channels: Number of channels of the set to predict.

        max_set_size: Maximum size of the set.

        iter: Number of iterations to run the DSPN algorithm for.

        lr: Learning rate of inner gradient descent in DSPN.
        """
        super().__init__()
        self.iters = iters
        self.lr = lr
        self.momentum = momentum
        self.set_size = set_size
        self.set_dim = set_dim
        self._set_0 = None
        self.projection = projection

        if learn_init_set:
            self._set_0  = nn.Parameter(0.1 * torch.randn(1, self.set_size, self.set_dim))

    def get_init_set(self, z):
        if self._set_0 is not None:
            return self._set_0.expand(z.size(0), -1, -1).requires_grad_()

        return 0.1 * torch.randn(z.size(0), self.set_size, self.set_dim, device=z.device, requires_grad=True)

    @torch.enable_grad()
    def forward(self, obj_fn, z, set_0=None):
        """
        Conceptually, DSPN simply turns the target_repr feature vector into a set.

        target_repr: Representation that the predicted set should match. FloatTensor of size (batch_size, repr_channels).
        Note that repr_channels can be different from self.channels.
        This can come from a set processed with the same encoder as self.encoder (auto-encoder), or a different
        input completely (normal supervised learning), such as an image encoded into a feature vector.
        """
        if set_0 is None:
            set_0 = self.get_init_set(z)
        # info used for loss computation
        intermediate_sets = [set_0]
        set_t = set_0.requires_grad_()

        # optimise repr_loss for fixed number of steps
        with torch.enable_grad():
            set_t_ = set_t.clone().detach()
            opt = higher.get_diff_optim(
                torch.optim.SGD([set_t_], lr=self.lr, momentum=self.momentum, nesterov=self.momentum > 0), 
                [set_t_], device=z.device)

            for i in range(self.iters):
                # how well does the representation matches the target
                repr_loss = obj_fn(z, set_t)
                set_t, = opt.step(repr_loss, params=[set_t])
                if self.projection is not None:
                    set_t = self.projection(set_t)
                intermediate_sets.append(set_t)
            output = set_t

        return output, torch.zeros_like(output)  # gradient info not directly available
