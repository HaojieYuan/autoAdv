import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

import cifar10_models
from aug_search import AUG_TYPE
from attacks import attack
import random
import pdb

class RewardCal():

    def __init__(self, dataset='imagenet'):
        assert dataset in ['imagenet', 'cifar10'], 'Unsupported searching dataset.'
        self.dataset = dataset
        if self.dataset == 'imagenet':
            imgnet_resize = transforms.Compose([transforms.Resize(342),
                                                transforms.CenterCrop(299),
                                                transforms.ToTensor()])
            self.dataset = torchvision.datasets.ImageNet('/home/haojieyuan/Data/ImageNet/ILSVRC_2012',
                                                         split='val', transform=imgnet_resize)
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])

            resnet18 = torchvision.models.resnet18(pretrained=True)
            alexnet = torchvision.models.alexnet(pretrained=True)
            squeezenet = torchvision.models.squeezenet1_0(pretrained=True)
            vgg16 = torchvision.models.vgg16(pretrained=True)
            densenet = torchvision.models.densenet161(pretrained=True)
            shufflenet = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
            mobilenet = torchvision.models.mobilenet_v2(pretrained=True)
            resnext = torchvision.models.resnext50_32x4d(pretrained=True)
            mnasnet = torchvision.models.mnasnet1_0(pretrained=True)

            self.models = [resnet18, alexnet, squeezenet, vgg16, densenet, shufflenet,
                           mobilenet, resnext, mnasnet]
            #self.models = [resnet18, alexnet, squeezenet, densenet, shufflenet,
            #               mobilenet, resnext, mnasnet]
        elif self.datset == 'cifar10':
            assert False, 'Not supported yet.'
            cifar_to_tensor = transforms.Compose([transforms.ToTensor()])
            self.dataset = torchvision.datasets.CIFAR10(root='/home/haojieyuan/Data/CIFAR_10_data', train=False,
                                       download=False, transform=cifar_to_tensor)
            self.normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                  std=[0.2023, 0.1994, 0.2010])
            resnet18 = cifar10_models.resnet18(pretrained=True)
            mobilenet_v2 = cifar10_models.mobilenet_v2(pretrained=True)
            googlenet = cifa10_models.googlenet(pretrained=True)

        else:
            assert False


        for model in self.models:
            model.eval()

        self.arrange_model_devices()

    def arrange_model_devices(self):
        '''
        self.models[0].cuda(1)  # resnet18
        self.models[1].cuda(2)  # alexnet
        self.models[2].cuda(0)  # squeezenet

        self.models[3].cuda(3)  # vgg16
        self.models[4].cuda(4)  # densenet
        self.models[5].cuda(5)  # shufflenet
        self.models[6].cuda(0)  # mobilenet
        self.models[7].cuda(6)  # resnext
        self.models[8].cuda(7)  # mnasnet
        '''
        '''
        self.models[3].cuda(9)  # densenet
        self.models[4].cuda(9)  # shufflenet
        self.models[5].cuda(8)  # mobilenet
        self.models[6].cuda(9)  # resnext
        self.models[7].cuda(9)  # mnasnet
        '''
        if self.dataset == 'imagenet':
            self.models[0].cuda(1)  # resnet18
            self.models[1].cuda(1)  # alexnet
            self.models[2].cuda(1)  # squeezenet
            self.models[3].cuda(2)  # vgg16
            self.models[4].cuda(2)  # densenet
            self.models[5].cuda(1)  # shufflenet
            self.models[6].cuda(1)  # mobilenet
            self.models[7].cuda(1)  # resnext
            self.models[8].cuda(2)  # mnasnet

        elif self.dataset == 'cifar10':
            pass

        else:
            assert False




    def randomrize_models(self):
        proxy_n = random.randint(1, len(self.models)-1)
        proxy_ids = []
        val_ids = []
        for i in range(proxy_n):
            new_id = random.randint(0, len(self.models)-1)
            while new_id in proxy_ids:
                new_id = random.randint(0, len(self.models)-1)

            proxy_ids.append(new_id)

        self.proxy_models = []
        self.eval_models = []

        for i in range(len(self.models)):
            if i in proxy_ids:
                self.proxy_models.append(self.models[i])
                #self.proxy_models.append(i)
            else:
                self.eval_models.append(self.models[i])
                #self.eval_models.append(i)



    def get_reward(self, policy, batch_size=8, shuffle=True, dataset_split=500):

        # Load data
        if dataset_split != 1:
            dataset_mask = random.sample(range(len(self.dataset)),
                                         len(self.dataset)//dataset_split)
            dataset = torch.utils.data.Subset(self.dataset, dataset_mask)
        else:
            dataset = self.dataset

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 shuffle=shuffle, drop_last=True)

        loss_fn = torch.nn.CrossEntropyLoss()
        rewards = []
        for img_batch, y in dataloader:

            img_batch_adv = attack(img_batch, self.proxy_models,
                                   aug_policy=policy, y=y, targeted=False,
                                   momentum_mu=1.0,  preprocess=self.normalize)
            # normalize for evaluation
            img_batch_adv = torch.stack([self.normalize(img_adv) for img_adv in img_batch_adv])

            with torch.no_grad():
                reward = self.eval(loss_fn, img_batch_adv, y)

            rewards.append(reward.detach().cpu().item())

        rewards_out = np.array(rewards).mean()

        return rewards_out


    def eval(self, loss_fn, imgs, gt_label):
        logits = None
        for model in self.eval_models:
            device = next(model.parameters()).device
            logit = model(imgs.to(device))
            logit = logit.detach().cpu()
            if logits is None:
                logits = logit
            else:
                logits = logits + logit

        logit = logits / len(self.eval_models)
        reward = loss_fn(logit, gt_label).detach().cpu()

        return reward






