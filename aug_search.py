import torch
import math

# TODO: color augmentation
AUG_TYPE = {0: 'resize_padding', 1: 'translation', 2: 'rotation', 
            3: 'gaussian_noise', 4: 'horizontal_flip', 5: 'vertical_flip'}

def augmentation(op_type, magnitude):
    ''' augmentation that capable of backward.
        returns a function that takes img tensor values from 0 to 1
    '''
    if op_type == 'resize_padding':
        def aug_func(img_tensor):
            img_w = img_tensor.shape[2]
            img_h = img_tensor.shape[3]
            w_modified = 2*int(0.01*magnitude*img_w)
            h_modified = 2*int(0.01*magnitude*img_h)
            img_tensor = torch.nn.functional.interpolate(img_tensor, 
                                                         [img_w-w_modified, img_h-h_modified])
            h_padding = h_modified//2
            w_padding = w_modified//2
            img_tensor = torch.nn.functional.pad(img_tensor, (h_padding, h_padding, w_padding, w_padding), 
                                                 mode='constant', value=0)
            return img_tensor

    elif op_type == 'translation':
        def aug_func(img_tensor):
            magnitude_ = magnitude-5 # 0to11 -> -5to5
            w_modified = 0.03*magnitude_
            h_modified = 0.03*magnitude_
            trans_M = torch.Tensor([[1., 0., w_modified],
                                    [0., 1., h_modified]])
            batch_size = img_tensor.shape[0]
            trans_M = trans_M.unsqueeze(0).repeat(batch_size, 1, 1)
            grid = torch.nn.functional.affine_grid(trans_M, img_tensor.shape)
            img_tensor = torch.nn.functional.grid_sample(img_tensor, grid.to(img_tensor.device))
            return img_tensor
    
    elif op_type == 'rotation':
        def aug_func(img_tensor):
            magnitude_ = magnitude-5 # 0to11 -> -5to5
            rot_deg = torch.tensor(math.pi*magnitude_/30.) # -pi/6 to pi/6
            rot_M = torch.Tensor([[torch.cos(rot_deg), -torch.sin(rot_deg), 0],
                                  [torch.sin(rot_deg), torch.cos(rot_deg), 0]])
            batch_size = img_tensor.shape[0]
            rot_M = rot_M.unsqueeze(0).repeat(batch_size, 1, 1)
            grid = torch.nn.functional.affine_grid(rot_M, img_tensor.shape)
            img_tensor = torch.nn.functional.grid_sample(img_tensor, grid.to(img_tensor.device))
            return img_tensor

    elif op_type == 'gaussian_noise':
        def aug_func(img_tensor):
            noise = torch.randn_like(img_tensor)
            img_tensor = img_tensor + noise * magnitude/60
            img_tensor = torch.clamp(img_tensor, 0, 1)
            return img_tensor

    elif op_type == 'horizontal_flip':
        def aug_func(img_tensor):
            return torch.flip(img_tensor, [3])

    elif op_type == 'vertical_flip':
        def aug_func(img_tensor):
            return torch.flip(img_tensor, [2])
    else:
        print(op_type)
        assert False, "Unknown augmentation type."
    
    return aug_func

