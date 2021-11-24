import torch
import torch.nn as nn
import torch.nn.functional as F


class iDSPN(nn.Module):
    def __init__(self, set_dim, set_size, learn_init_set, inner_obj, optim_f, optim_iters, grad_clip=None, projection=None):
        super().__init__()
        self.inner_obj = inner_obj
        self.iters = optim_iters
        self.optim_f = optim_f
        self.projection = projection
        self.grad_clip = grad_clip
        self.set_dim = set_dim
        self.set_size = set_size
        self._set_0 = None

        if learn_init_set:
            self._set_0  = nn.Parameter(0.1 * torch.randn(1, self.set_size, self.set_dim))

    def get_init_set(self, z):
        if self._set_0 is not None:
            return self._set_0.expand(z.size(0), -1, -1).requires_grad_()

        return 0.1 * torch.randn(z.size(0), self.set_size, self.set_dim, device=z.device)

    def forward(self, obj_fn, z, set_0=None):
        """
        Args:
            obj_fn: (z, set_t) -> scalar
            z: vector; inputt encoding
            set_0 (optional): initial set that iDSPN refines
        """
        if set_0 is None:
            set_0 = self.get_init_set(z)
        _obj_fn = lambda set_t: obj_fn(z, set_t)
        return ImplicitSolver.apply(_obj_fn, self.optim_f, self.iters, self.grad_clip, self.projection, set_0, z, *self.inner_obj.parameters())


class Objective(nn.Module):
    def forward(self, target_repr, set_t, reference_set=None):
        raise NotImplementedError


class MSEObjective(Objective):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, target_repr, set_t):
        # compute representation of current set_t
        predicted_repr = self.encoder(set_t)
        # how well does the representation match the target
        repr_loss = 0.5 * F.mse_loss(
            predicted_repr, target_repr, reduction='none'
        ).sum(dim=0).mean()

        return repr_loss

class MSEObjectiveRegularized(MSEObjective):
    def forward(self, target_repr, set_t, reference_set):
        repr_loss = super().forward(target_repr, set_t)
        regularizer = 0.5 * F.mse_loss(set_t, reference_set, reduction='none').sum(dim=0).mean()
        repr_loss = repr_loss + 0.1 * regularizer
        return repr_loss


class MSEObjectiveCatInput(MSEObjective):
    def forward(self, target_repr, set_t, input_set):
        set_t = torch.cat([set_t, input_set], dim=2)
        return super().forward(target_repr, set_t)


class ImplicitSolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, objective_fn, optim_f, iters, grad_clip, projection, set_0, *params):
        # make sure that all parameters passed are used in the computation graph
        # otherwise, you have to set_t allow_unused=True in the autograd.grad call in backwards 

        # if regularization is used in the objective, assumes that the set_t to start with is the one to regularize with 
        # this doesn't hold when dspn iters are split into multiple forwards and the intention is to regularize wrt idspn.starting_set
        set_t = set_0.clone().detach().requires_grad_(True)
        if projection is not None:
            set_t.data = projection(set_t.data)

        optimizer = optim_f([set_t])
        with torch.enable_grad():
            # iterate n - 1 steps
            for i in range(iters - 1):
                loss = objective_fn(set_t)
                set_t.grad, = torch.autograd.grad(loss, set_t)
                set_t.grad = clip_gradient(set_t.grad, max_norm=grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                if projection is not None:
                    set_t.data = projection(set_t.data)
            # iterate last step
            # we don't want the optimizer to override our set_t with in-place modifications, so we do this one separately
            set_opt = set_t.clone().detach().requires_grad_(True)
            loss = objective_fn(set_opt)
            set_grad, = torch.autograd.grad(loss, set_opt, create_graph=True)
            set_t.grad = clip_gradient(set_grad.clone(), max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            if projection is not None:
                set_t.data = projection(set_t.data)
                set_grad = projection(set_opt - set_grad) - set_opt
            else:
                set_grad = -set_grad

        ctx.save_for_backward(set_opt, set_grad, set_0, *params)
        return set_t, set_grad.clone()
    
    @staticmethod
    def backward(ctx, output_grad, set_grad_grad):
        set_opt, set_grad, *inputs = ctx.saved_tensors
        n_none = 5

        total_grad = output_grad # - set_grad_grad
        # important to have the same order as given to forward
        # only need to differentiate wrt inputs that actually require grads
        inputs_to_differentiate = [(input, i) for i, (input, needs_grad) in enumerate(zip(inputs, ctx.needs_input_grad[n_none:])) if needs_grad]

        # normal conjugate gradient
        # def HVP(x):
        #     with torch.enable_grad():
        #         return torch.autograd.grad(set_grad, set_opt, retain_graph=True, grad_outputs=x)[0]
        # u = conjugate_gradient(HVP, torch.zeros_like(total_grad), total_grad, iters=10)

        # regularized conjugate gradient
        # def HVP_regularized(x):
        #     with torch.enable_grad():
        #         return torch.autograd.grad(set_grad, set_opt, retain_graph=True, grad_outputs=x)[0] / 100 + x
        # u = conjugate_gradient(HVP_regularized, total_grad, total_grad, iters=3)  # use this line instead for conjugate gradient approach
        
        # approximate implicit diff
        u = total_grad
        
        # in certain cases with a starting set_t, the following line needs retain_graph=True added
        with torch.enable_grad():
            grads = torch.autograd.grad(set_grad, [x[0] for x in inputs_to_differentiate], u)

        # always need to return something for all inputs, so put the grads back in their corresponding position
        padded_grads = [None for _ in range(len(inputs))]
        for g, (_, i) in zip(grads, inputs_to_differentiate):
            padded_grads[i] = g

        return (*[None]*n_none, *padded_grads)


def clip_gradient(grads, norm_type=2., max_norm=2.):
    if max_norm is None:
        return grads
    grad_norm = grads.detach().norm(norm_type, dim=list(range(1, grads.ndim)), keepdim=True)
    clip_coef = max_norm / (grad_norm + 1e-6)
    clip_coef = clip_coef.clamp(0., 1.)
    grads = grads * clip_coef
    return grads


def conjugate_gradient(hvp, x_init, b, iters=3):
    x = x_init
    r = b - hvp(x)
    p = r
    bdot = lambda a, b: torch.einsum('nsc, nsc -> n', a, b).clamp(min=1e-37)
    for i in range(iters):
        Ap = hvp(p)
        alpha = bdot(r, r) / bdot(p, Ap)
        alpha = alpha.unsqueeze(1).unsqueeze(2)
        x = x + alpha * p
        r_new = r - alpha * Ap
        beta = bdot(r_new, r_new) / bdot(r, r)
        beta = beta.unsqueeze(1).unsqueeze(2)
        p = r_new + beta * p
        r = r_new
    return x


class ProjectSimplexModule(nn.Module):
    def __init__(self, value, dim=2):
        super().__init__()
        self.value = value
        self.dim = dim

    def forward(self, x):
        return self.value * ProjectSimplex.apply(x / self.value, self.dim)


class ProjectSimplex(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim=2):
        x_proj = projection_unit_simplex(x, dim=dim)
        ctx.save_for_backward(x, x_proj)
        ctx.dim = dim
        return x_proj

    @staticmethod
    def backward(ctx, x_proj_grad):
        x, x_proj = ctx.saved_tensors
        out_grad = projection_unit_simplex_jvp(x, x_proj, x_proj_grad, ctx.dim)
        return out_grad, None


def unsqueeze_like(x, target, match_dim):
    shape = [1]*target.ndim
    shape[match_dim] = -1
    return x.reshape(*shape)


def batched_idx(idx, dim):
    set_dim = 2 if dim==1 else 1 # 1
    bid = torch.arange(idx.size(0), device=idx.device).repeat_interleave(idx.size(set_dim))
    sid = torch.arange(idx.size(set_dim), device=idx.device).repeat(idx.size(0))
    ret = [bid, None, None]
    ret[dim] = idx.flatten()
    ret[2 if dim==1 else 1] = sid.flatten()
    return ret


def projection_unit_simplex(x, dim):
    s = 1.0
    n_features = x.shape[dim]
    u, _ = torch.sort(x, dim=dim, descending=True)
    cssv = torch.cumsum(u, dim=dim) - s
    ind = torch.arange(n_features, device=x.device) + 1
    cond = u - cssv / unsqueeze_like(ind, cssv, dim) > 0
    idx = torch.count_nonzero(cond, dim=dim)
    threshold = cssv[batched_idx(idx - 1, dim=dim)].reshape(idx.shape) / idx.to(x.dtype)
    return torch.relu(x - threshold.unsqueeze(dim))


def projection_unit_simplex_jvp(x, x_proj, x_proj_grad, dim):
    supp = x_proj > 0
    card = torch.count_nonzero(supp, dim=dim).unsqueeze(dim)
    supp = supp.to(x_proj_grad.dtype)
    prod = supp * x_proj_grad
    tangent_out = prod - (prod.sum(dim=dim, keepdim=True) / card) * supp
    return tangent_out