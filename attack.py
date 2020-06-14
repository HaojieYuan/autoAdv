import torch
import numpy as np
import math

def attack(img_batch, model, aug_list=None, type='iterative', momentum_mu=None,
           y=None, eps=5, eps_iter=2, nb_iter=3, ord=2, clip_min=0, clip_max=1):
    
    device = img_batch.device
    if ord not in [np.inf, 1, 2]:
        raise ValueError("Norm order must be either np.inf or 2.")
    l2 = (clip_max-clip_min)/255 * math.sqrt(img_batch.shape[1]*img_batch.shape[2]*img_batch.shape[3])
    if ord == 2:
        eps = eps*l2
        eps_iter = eps_iter*l2
    
    x0 = img_batch.clone().detach().to(torch.float).requires_grad_(False)
    if y is None:
        _, y = torch.max(model(x0), 1)
        targeted = False
    else:
        targeted = True

    x = img_batch.clone().detach().to(torch.float).requires_grad_(True)
    eta = torch.zeros_like(x).to(device)
    eta = clip_eta(eta, ord, eps)
    adv_x = x + eta
    if clip_min is not None or clip_max is not None:
        adv_x = torch.clamp(adv_x, clip_min, clip_max)
    
    i = 0
    if momentum_mu is not None:
        momentum = eta.clone().detach()
    loss_fn = torch.nn.CrossEntropyLoss()
    while i < nb_iter:
        adv_x_tmp = adv_x.clone().detach().to(torch.float).requires_grad_(True)    
        
        adv_x_list = [adv_x_tmp]         
        if aug_list is not None:
            adv_x_list.extend([aug_func[1](aug_func[0](adv_x_tmp)) for aug_func in aug_list['augs']])
            weights = aug_list['weights']
        else:
            weights = [1]

        loss = torch.tensor(0.).to(device)
        for j, k in zip(adv_x_list, weights):
            loss = loss + k * loss_fn(model(j), y)
        if not targeted:
            loss = -loss
        adv_x_tmp = single_step(loss, adv_x_tmp, eps_iter, ord, clip_min=clip_min, clip_max=clip_max)
        adv_x = adv_x_tmp.clone().detach()
        del adv_x_tmp

        eta = adv_x - x0
        eta = clip_eta(eta, ord, eps)
        if momentum_mu is not None:
            momentum = momentum_mu * momentum + eta
            eta = clip_eta(momentum, ord, eps)
        adv_x = x0 + eta
        model.zero_grad()

        if clip_min is not None or clip_max is not None:
            adv_x = torch.clamp(adv_x, clip_min, clip_max)
        i += 1
    
    return adv_x





def single_step(loss, adv_x, eps, ord, clip_min=None, clip_max=None):
    loss = loss.sum()
    loss.backward()
    optimal_perturbation = optimize_linear(adv_x.grad, eps, ord)
    adv_x = adv_x - optimal_perturbation
    if (clip_min is not None) or (clip_max is not None):
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    return adv_x
        

def optimize_linear(grad, eps, ord=np.inf):
    """
    Solves for the optimal input to a linear function under a norm constraint.

    Optimal_perturbation = argmax_{eta, ||eta||_{ord} < eps} dot(eta, grad)

    :param grad: Tensor, shape (N, d_1, ...). Batch of gradients
    :param eps: float. Scalar specifying size of constraint region
    :param ord: np.inf, 1, or 2. Order of norm constraint.
    :returns: Tensor, shape (N, d_1, ...). Optimal perturbation
    """

    red_ind = list(range(1, len(grad.size())))
    avoid_zero_div = torch.tensor(1e-12, dtype=grad.dtype, device=grad.device)
    if ord == np.inf:
        # Take sign of gradient
        optimal_perturbation = torch.sign(grad)
    elif ord == 1:
        abs_grad = torch.abs(grad)
        sign = torch.sign(grad)
        red_ind = list(range(1, len(grad.size())))
        abs_grad = torch.abs(grad)
        ori_shape = [1]*len(grad.size())
        ori_shape[0] = grad.size(0)

        max_abs_grad, _ = torch.max(abs_grad.view(grad.size(0), -1), 1)
        max_mask = abs_grad.eq(max_abs_grad.view(ori_shape)).to(torch.float)
        num_ties = max_mask
        for red_scalar in red_ind:
            num_ties = torch.sum(num_ties, red_scalar, keepdim=True)
        optimal_perturbation = sign * max_mask / num_ties
        # TODO integrate below to a test file
        # check that the optimal perturbations have been correctly computed
        opt_pert_norm = optimal_perturbation.abs().sum(dim=red_ind)
        assert torch.all(opt_pert_norm == torch.ones_like(opt_pert_norm))
    elif ord == 2:
        square = torch.max(
            avoid_zero_div,
            torch.sum(grad ** 2, red_ind, keepdim=True)
            )
        optimal_perturbation = grad / torch.sqrt(square)
        # TODO integrate below to a test file
        # check that the optimal perturbations have been correctly computed
        opt_pert_norm = optimal_perturbation.pow(2).sum(dim=red_ind, keepdim=True).sqrt()
        one_mask = (square <= avoid_zero_div).to(torch.float) * opt_pert_norm + \
                (square > avoid_zero_div).to(torch.float)
        assert torch.allclose(opt_pert_norm, one_mask, rtol=1e-05, atol=1e-08)
    else:
        raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                    "currently implemented.")

    # Scale perturbation to be the solution for the norm=eps rather than
    # norm=1 problem
    scaled_perturbation = eps * optimal_perturbation
    return scaled_perturbation

def clip_eta(eta, ord, eps):
    """
    PyTorch implementation of the clip_eta in utils_tf.

    :param eta: Tensor
    :param ord: np.inf, 1, or 2
    :param eps: float
    """
    if ord not in [np.inf, 1, 2]:
        raise ValueError('ord must be np.inf, 1, or 2.')

    avoid_zero_div = torch.tensor(1e-12, dtype=eta.dtype, device=eta.device)
    reduc_ind = list(range(1, len(eta.size())))
    if ord == np.inf:
        eta = torch.clamp(eta, -eps, eps)
    else:
        if ord == 1:
            # TODO
            # raise NotImplementedError("L1 clip is not implemented.")
            norm = torch.max(
                avoid_zero_div,
                torch.sum(torch.abs(eta), dim=reduc_ind, keepdim=True)
            )
        elif ord == 2:
            norm = torch.sqrt(torch.max(
                avoid_zero_div,
                torch.sum(eta ** 2, dim=reduc_ind, keepdim=True)
            ))
        factor = torch.min(
            torch.tensor(1., dtype=eta.dtype, device=eta.device),
            eps / norm
            )
        eta *= factor
    return eta
