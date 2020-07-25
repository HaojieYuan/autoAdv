import torch
import numpy as np
import math
import random
from aug_search import augmentation, AUG_TYPE
import pdb

def attack(img_batch, models, aug_policy=None, momentum_mu=None, targeted=False, preprocess=None,
           y=None, eps=16, eps_iter=1.6, nb_iter=10, ord=np.inf, clip_min=0, clip_max=1):

    if ord not in [np.inf, 2]:
        raise ValueError("Norm order must be either np.inf or 2.")

    # Normalized constraint.
    l2 = (clip_max-clip_min)/255 * math.sqrt(img_batch.shape[1]*img_batch.shape[2]*img_batch.shape[3])
    if ord == 2:
        eps = eps*l2
        eps_iter = eps_iter*l2
    elif ord == np.inf:
        eps = eps/255.
        eps_iter = eps_iter/255.


    x0 = img_batch.clone().detach().to(torch.float).requires_grad_(False)

    if y is None:
        _, y = torch.max(models[0](x0), 1)

    if aug_policy is not None:
        aug_num = len(aug_policy)
        y = torch.cat(aug_num*[y], dim=0)             # [Batchsize*aug_type, 1]
        weights = get_weights(aug_policy)
        # [Batchsize*aug_type]
        weights = torch.cat(x0.shape[0]*[weights.reshape(-1, 1)], dim=1).reshape(-1)
    else:
        weights = torch.tensor((x0.shape[0])*[1]) # [Bs,]

    x = img_batch.clone().detach().to(torch.float).requires_grad_(False)
    eta = torch.zeros_like(x)
    eta = clip_eta(eta, ord, eps)
    adv_x = x + eta
    if clip_min is not None or clip_max is not None:
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    if momentum_mu is not None:
        momentum = eta.clone().detach()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')


    # main loop
    for i in range(nb_iter):
        adv_x = adv_x.clone().detach().requires_grad_(False)

        # models are saved on different devices, thus we get gradients separately
        gradients = []
        for model in models:
            device = next(model.parameters()).device
            adv_x = adv_x.clone().detach().to(device).requires_grad_(True)
            weights = weights.to(device)

            # get augmented data
            if aug_policy is not None:
                # adv_x_list: [Batch_size*aug_type, 3, 299, 299]
                adv_x_list = torch.cat(augment(adv_x, aug_policy), dim=0)
            else:
                adv_x_list = adv_x


            loss = (weights * loss_fn(model(adv_x_list), y.to(device))).mean()
            if not targeted:
                loss = - loss



            loss.backward()

            #gradient = adv_x_list.grad # [Batch_size*aug_type,3,299,299]
            #gradient = (gradient.reshape(-1, *(x0.shape))).mean(dim=0).detach().cpu() # [Batch_size,3,299,299]
            gradient = adv_x.grad # [Batch_size, 3, 299, 299]
            gradients.append(gradient.detach().cpu())

            del adv_x_list
            model.zero_grad()
            adv_x.grad.data.zero_()


        gradient = torch.stack(gradients).mean(0)


        # single step
        if momentum_mu is not None:
            normalized_perturbation = optimize_linear(gradient, 1, ord)
            momentum = momentum_mu * momentum + normalized_perturbation
            momentum = optimize_linear(momentum, eps_iter, ord)
            optimal_perturbation = momentum
        else:
            optimal_perturbation = optimize_linear(gradient, 1, ord)

        adv_x = adv_x.detach().cpu()
        adv_x = adv_x - optimal_perturbation
        if clip_min is not None or clip_max is not None:
            adv_x = torch.clamp(adv_x, clip_min, clip_max)


        # clip again to make sure accumulated perturbation size is under constraint.
        eta = adv_x - x0
        eta = clip_eta(eta, ord, eps)
        adv_x = x0 + eta

        if clip_min is not None or clip_max is not None:
            adv_x = torch.clamp(adv_x, clip_min, clip_max)

    return adv_x.detach().cpu().requires_grad_(False)



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


def augment(x, aug_policy):
    x_out = []
    for sub_policy in aug_policy:
        aug_type  = AUG_TYPE[sub_policy[0]]
        aug_prob  = sub_policy[2] + 1  # 1~10
        aug_range = sub_policy[3] + 1 # 1~10

        if random.uniform(0, 10) < aug_prob:
            aug_mag = random.choice(range(0, int(aug_range)+1))
            aug_x = augmentation(x, aug_type, aug_mag)
        else:
            aug_x = x
        x_out.append(aug_x)

    return x_out



def get_weights(aug_policy):
    weights = [policy[1] for policy in aug_policy]
    weights = torch.tensor(weights).to(torch.float)

    # Normalize
    weights = weights/weights.sum()

    return weights

