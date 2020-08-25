import random
random.seed(13)

import argparse
import pdb

'''
parser = argparse.ArgumentParser(description='Nothing')
parser.add_argument('--listf', default='123')
parser.add_argument('--outf', default='456')
args = parser.parse_args()

#in_list = '/home/haojieyuan/Data/oxfordFlowers/test_list.txt'

pdb.set_trace()
'''

#in_list = args.listf
in_list = "/home/haojieyuan/Data/caltech101/test_list.txt"
out_list = "/home/haojieyuan/Data/caltech101/eval_gt_labels.txt"
f = open(in_list)

img_list = []
id_list = []
for line in f:
    img_path, id_ = line.strip().split(' ')
    img_list.append(img_path)
    id_list.append(int(id_) -1)

f.close()

assert len(img_list)==len(id_list)
total_num = len(img_list)

ordered_list = list(range(total_num))
shuffle_list = random.sample(ordered_list, 1000)

picked_img_list = []
picked_id_list = []

for i in shuffle_list:
    picked_img_list.append(img_list[i])
    picked_id_list.append(id_list[i])

#out_list = args.outf
t = open(out_list, 'w')

for img_path, id_ in zip(picked_img_list, picked_id_list):
    t.write(img_path+' '+str(id_)+'\n')

t.close()