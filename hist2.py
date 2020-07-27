import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

import pdb


class GaussianHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(GaussianHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)

    '''
    def forward(self, x):
        x = x.reshape(-1)
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.exp(-0.5*(x/self.sigma)**2) / (self.sigma * np.sqrt(np.pi*2)) * self.delta
        x = x.sum(dim=1)
        return x
    '''
    def forward(self, x):
        x = x.reshape(-1)
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.sigmoid(self.sigma * (x + self.delta/2)) - torch.sigmoid(self.sigma * (x - self.delta/2))
        x = x.sum(dim=1)
        return x

if __name__ == '__main__':
    #data = torch.randn(1000)
    im = Image.open('clean.png')
    to_tensor = transforms.Compose([transforms.ToTensor()])
    x = to_tensor(im)
    data = x[0,:,:]

    data = data * 255.
    hist1 = torch.histc(data, bins=256, min=0, max=255)
    print(hist1)


    gausshist = GaussianHistogram(bins=256, min=0, max=100, sigma=6)

    data.requires_grad = True
    hist2 = gausshist(data)
    print(hist2)

    print((hist1-hist2)/hist1)

    hist2.sum().backward()
    print(data.grad)
