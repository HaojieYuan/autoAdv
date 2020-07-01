import torch
import torchvision
import torchvision.transforms as transforms
from attack import attack
from aug_search import AUG_TYPE, augmentation
from tqdm import tqdm
import random
import cifar10_models
import pdb
random.seed(1)

def get_rewards(actions, device_id=1):
    # a batch of actions
    # calculate them separately.
    actions_op = actions['op']
    actions_weight = actions['weight']
    rewards = []
    for policy, weight in zip(actions_op, actions_weight):
        # policy [5, 2, 2]
        # weight [6]
        aug_list = []
        for sub_policy in policy:
            # sub_policy [2, 2]
            sub_aug_list = []
            for operation in sub_policy:
                # operation [2]
                op_type, op_mag = operation
                op_type = op_type.detach().cpu().item()
                op_mag = op_mag.detach().cpu().item()
                op_type = AUG_TYPE[op_type]
                sub_aug_list.append(augmentation(op_type, op_mag))
            aug_list.append(sub_aug_list)
        weight = weight.to(torch.float)
        #weight = torch.nn.functional.softmax(weight, dim=-1).detach().cpu().tolist()
        weight = (weight/weight.sum()).detach().cpu().tolist()
        aug = {'augs':aug_list,'weights':weight}
        reward = get_reward(aug, device_id=device_id)
        rewards.append(reward)
    return torch.Tensor(rewards).unsqueeze(1)
    

def get_reward(aug_list, batch_size=8, device_id=1, dataset_name='cifar10'):
    data_loader, proxy_model, eval_model, test_model = load_dataset(dataset_name, batch_size)
    proxy_model = proxy_model.cuda(device_id)
    eval_model = eval_model.cuda(device_id)
    loss_fn = torch.nn.CrossEntropyLoss()
    reward = 0
    for img_batch, y in data_loader:
        img_batch = img_batch.cuda(device_id)
        y = y.cuda(device_id)
        img_batch_adv = attack(img_batch, proxy_model, aug_list=aug_list)
        with torch.no_grad():
            logit_adv = eval_model(img_batch_adv)
        reward = reward + loss_fn(logit_adv, y).detach().cpu().item()
    del proxy_model
    del eval_model
    del test_model

    return reward 




def load_dataset(dataset_name, batch_size, shuffle=True, Full=False):
    cifar10_root = '/home/haojieyuan/Data/CIFAR_10_data'
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(cifar10_root, train=True, 
                                               transform=transform, download=True)
        dataset_split = 30
        #dataset_split = 500
        proxy_model = cifar10_models.resnet18(pretrained=True)
        eval_model = cifar10_models.inception_v3(pretrained=True)
        test_model = cifar10_models.mobilenet_v2(pretrained=True)
    else:
        pass
    
    if not Full:
        dataset_mask = random.sample(range(len(dataset)), len(dataset)//dataset_split)
        dataset = torch.utils.data.Subset(dataset, dataset_mask)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    proxy_model.eval()
    eval_model.eval()
    test_model.eval()

    return dataloader, proxy_model, eval_model, test_model