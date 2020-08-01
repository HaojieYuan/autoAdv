import torch
import pdb
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
import math

def equalize(img_tensor):
    img_tensor = img_tensor * 255. #0~1 to 0~255

    def scale_channel(im, c):
        im = im[c, :, :]
        histo = torch.histc(im, bins=256, min=0, max=255).detach()
        nonzero_histo = torch.reshape(histo[histo!=0], [-1])
        step = (torch.sum(nonzero_histo)-nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            lut = (torch.cumsum(histo, 0)) + (step//2)//step

            lut = torch.cat([torch.zeros(1).to(im.device), lut[:-1]])

            return torch.clamp(lut, 0, 255)

        if step == 0:
            result = im
        else:
            result = torch.gather(build_lut(histo, step), 0, im.flatten().long())
            result = result.reshape_as(im).to(torch.float32)

        return result/255.

    res = []
    for image in img_tensor:
        scaled_image = torch.stack([scale_channel(image, i) for i in range(len(image))])
        res.append(scaled_image)

    return torch.stack(res)

def resize(img_tensor):
    magnitude = 5
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


def translation(img_tensor):
    magnitude = 5
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

def rotation(img_tensor):
    magnitude = 5
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

def solarize(img_tensor):
    magnitude = 5
    solarize_threshold = 1.0 - 0.09*magnitude
    return torch.where(img_tensor < solarize_threshold, img_tensor, 1.0-img_tensor)


def invert(img_tensor):
    return 1.0 - img_tensor


def scaling(img_tensor):
    magnitude = 5

    img_tensor = img_tensor*2.0 -1.0
    magnitude = 1.0 - 0.1*magnitude
    img_tensor = img_tensor * magnitude
    img_tensor = (img_tensor + 1.0)/2.0

    return img_tensor




def trans_img_tensor(x, batch_dim=True):
    if batch_dim:
        x_numpy = x.detach().cpu().squeeze(0).permute(1,2,0).numpy()
    else:
        x_numpy = x.detach().cpu().permute(1,2,0).numpy()
    x_numpy = x_numpy*255
    im = Image.fromarray(np.uint8(x_numpy))
    return im


to_tensor = transforms.Compose([transforms.ToTensor()])
x1 = Image.open('clean1.png')
x2 = Image.open('clean2.png')
img_batch = torch.stack([to_tensor(x1), to_tensor(x2)])

img_batch.requires_grad_()
equalized_img = equalize(img_batch)
out1 = trans_img_tensor(equalized_img[0], batch_dim=False)
out2 = trans_img_tensor(equalized_img[1], batch_dim=False)

out1.save('out1.png')
out2.save('out2.png')

sum_ = equalized_img.sum()
sum_.backward()
print(img_batch.grad.sum())

