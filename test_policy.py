
import torch
from aug_search import AUG_TYPE, augmentation
import cifar10_models
from reward_calc import load_dataset
from tqdm import tqdm
from attack import attack
import random
random.seed(1)



op_list  =  [[['horizontal_flip',6],['resize_padding',8]],
            [['horizontal_flip',9],['gaussian_noise',4]],
            [['horizontal_flip',7],['resize_padding',1]],
            [['vertical_flip',10],['horizontal_flip',9]],
            [['resize_padding',4],['resize_padding',2]]]
weight = torch.Tensor([6, 0, 1, 8, 4, 5])
weight = torch.nn.functional.softmax(weight, dim=-1).detach().cpu().tolist()

'''
augs = [AUG_TYPE[key] for key in AUG_TYPE.keys()]

op_list = []
for i in range(5):
    sub_op_list = []
    for j in range(2):
        sub_op_list.append([random.choice(augs), random.randint(0,11)])
    op_list.append(sub_op_list)
'''
#op_list = [[['horizontal_flip',6],['horizontal_flip',5]]]
#print(op_list)
def get_aug_list(op_list):
    aug_list = []
    for i in op_list:
        sub_list = []
        for j in i:
            sub_list.append(augmentation(j[0], j[1]))
        aug_list.append(sub_list)
    return aug_list

aug_list = get_aug_list(op_list)
aug = {'augs':aug_list,'weights':weight}

def test(aug_list, batch_size=8, device_id=1, dataset_name='cifar10'):
    data_loader, proxy_model, eval_model, test_model = load_dataset(dataset_name, batch_size, Full=True)
    proxy_model = proxy_model.cuda(device_id)
    test_model = test_model.cuda(device_id)

    loss_fn = torch.nn.CrossEntropyLoss()

    clean_correct = 0
    adv_incorrect = 0
    for img_batch, y in tqdm(data_loader):

        '''
        mag = random.randint(0,11)
        op_list = [[['gaussian_noise',0],['rotation',mag]],
                   [['gaussian_noise',0],['gaussian_noise',0]],]
        aug_list = get_aug_list(op_list)
        '''


        img_batch = img_batch.cuda(device_id)
        y = y.cuda(device_id)
        img_batch_adv = attack(img_batch, proxy_model, aug_list=aug)
        with torch.no_grad():
            clean_outputs = test_model(img_batch)
            adv_outputs = test_model(img_batch_adv)
            _, adv_predictions = torch.max(adv_outputs.data, 1)
            _, clean_predictions = torch.max(clean_outputs.data, 1)
            clean_result = (clean_predictions == y)
            adv_result = (adv_predictions != y)
            clean_correct += clean_result.sum().item()
            adv_incorrect += (adv_result * clean_result).sum().item()
    
    attack_succ_rate = 100.*adv_incorrect/clean_correct
    print("Attack Success rate: %4f"%(attack_succ_rate))
 
#aug_list = None
test(aug_list, device_id=1)