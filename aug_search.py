import torch
import math

import random

# TODO: color augmentation
AUG_TYPE = {0: 'resize_padding', 1: 'translation', 2: 'rotation',
            3: 'gaussian_noise', 4: 'horizontal_flip', 5: 'vertical_flip',
            6: 'scaling', 7: 'invert', 8: 'solarize', 9: 'equalize'}

def augmentation(img_tensor, op_type, magnitude):
    ''' augmentation that capable of backward.
        with given magnitude, augmentations are done with random directions.
        Inputs: img_tensor range from 0 to 1,
                operation type in str description,
                magnitude range from 0 to 9.
        Return: augmented img tensor.
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

    elif op_type == 'scaling':
        # Refer to ICLR 2020 paper:
        # "NESTEROV ACCELERATED GRADIENT AND SCALE INVARIANCE FOR ADVERSARIAL ATTACKS"
        # https://arxiv.org/abs/1908.06281
        # And its implementation:
        # https://github.com/JHL-HUST/SI-NI-FGSM/blob/master/si_mi_fgsm.py
        # In its implementation, the scaling op is performed on image scaled to [-1, 1].
        # We don't know if such op is resonable because it is actually reduing contrast
        # to a biased mean [125, 125, 125]. However, we still follow such implementation here.
        # Meanwhile, the original implementation uses 1, 1/2, 1/4, 1/8, 1/16
        # which is actually 1, 0.5, 0.25, 0.125, 0.0625.
        # Here we make search range roughly contains such scales, 0.1 to 1.0
        img_tensor = img_tensor*2.0 -1.0
        magnitude = 1.0 - 0.1*magnitude
        img_tensor = img_tensor * magnitude
        img_tensor = (img_tensor + 1.0)/2.0

        return img_tensor

    elif op_type == 'invert':
        return 1.0 - img_tensor

    elif op_type == 'solarize':
        solarize_threshold = 256 - 25.6*magnitude
        return torch.where(img_tensor < solarize_threshold, img_tensor, 1.0-img_tensor)

    elif op_type == 'equalize':
        # code taken from https://github.com/kornia/
        img_tensor = img_tensor * 255. #0~1 to 0~255

        def scale_channel(im, c):
            im = im[c, :, :]
            histo = torch.histc(im, bins=256, min=0, max=255)
            nonzero_histo = torch.reshape(histo[histo!=0], [-1])
            step = (torch.sum(nonzero_histo)-nonzero_histo[-1]) // 255

            def build_lut(histo, step):
                lut = (torch.cumsum(histo, 0)) + (step//2)//step
                lut = torch.cat([torch.zeros(1), lut[:-1]])

                return torch.clamp(lut, 0, 255)

            if step == 0:
                result = im
            else:
                result = torch.gather(build_lut(histo, step), 0, im.flatten().long())
                result = result.reshape_as(im)

            return result/255.

        res = []
        for image in img_tensor:
            scaled_image = torch.stack([scale_channel(image, i)] for i in range(len(image)))
            res.append(scaled_image)

        return torch.stack(res)


    else:
        print(op_type)
        assert False, "Unknown augmentation type."
        return img_tensor


