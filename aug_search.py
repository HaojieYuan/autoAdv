import torch
import math

import random

# TODO: color augmentation
AUG_TYPE = {0: 'resize_padding', 1: 'translation', 2: 'rotation',
            3: 'gaussian_noise', 4: 'horizontal_flip', 5: 'vertical_flip'}

def augmentation(img_tensor, op_type, magnitude):
    ''' augmentation that capable of backward.
        with given magnitude, augmentations are done with random directions.
        returns augmented img tensor
    '''
    if op_type == 'resize_padding':
        img_w = img_tensor.shape[2]
        img_h = img_tensor.shape[3]
        w_modified = 2*int(0.01*magnitude*img_w)
        h_modified = 2*int(0.01*magnitude*img_h)
        img_tensor = torch.nn.functional.interpolate(img_tensor,
                                                     [img_w-w_modified, img_h-h_modified])

        h_padding_t = random.choice(range(0, h_modified+1))
        h_padding_b = h_modified - h_padding_t
        w_padding_l = random.choice(range(0, w_modified+1))
        w_padding_r = w_modified - w_padding_l
        #h_padding = h_modified//2
        #w_padding = w_modified//2
        img_tensor = torch.nn.functional.pad(img_tensor, (h_padding_t, h_padding_b, w_padding_l, w_padding_r),
                                             mode='constant', value=0)
        return img_tensor

    elif op_type == 'translation':
        w_direction = random.choice([-1, 1])
        h_direction = random.choice([-1, 1])

        #magnitude_ = magnitude-5 # 0to11 -> -5to5
        magnitude_ = magnitude
        w_modified = w_direction*0.02*magnitude_
        h_modified = h_direction*0.02*magnitude_
        trans_M = torch.Tensor([[1., 0., w_modified],
                                [0., 1., h_modified]])
        batch_size = img_tensor.shape[0]
        trans_M = trans_M.unsqueeze(0).repeat(batch_size, 1, 1)
        grid = torch.nn.functional.affine_grid(trans_M, img_tensor.shape)
        img_tensor = torch.nn.functional.grid_sample(img_tensor, grid.to(img_tensor.device))
        return img_tensor

    elif op_type == 'rotation':
        rot_direction = random.choice([-1, 1])
        #magnitude_ = magnitude-5 # 0to11 -> -5to5
        magnitude_ = magnitude
        rot_deg = torch.tensor(rot_direction*math.pi*magnitude_/60.) # -pi/6 to pi/6
        rot_M = torch.Tensor([[torch.cos(rot_deg), -torch.sin(rot_deg), 0],
                              [torch.sin(rot_deg), torch.cos(rot_deg), 0]])
        batch_size = img_tensor.shape[0]
        rot_M = rot_M.unsqueeze(0).repeat(batch_size, 1, 1)
        grid = torch.nn.functional.affine_grid(rot_M, img_tensor.shape)
        img_tensor = torch.nn.functional.grid_sample(img_tensor, grid.to(img_tensor.device))
        return img_tensor

    elif op_type == 'gaussian_noise':
        noise = torch.randn_like(img_tensor)
        img_tensor = img_tensor + noise * magnitude/60
        img_tensor = torch.clamp(img_tensor, 0, 1)
        return img_tensor

    elif op_type == 'horizontal_flip':
        return torch.flip(img_tensor, [3])

    elif op_type == 'vertical_flip':
        return torch.flip(img_tensor, [2])
    else:
        print(op_type)
        assert False, "Unknown augmentation type."
        return img_tensor


