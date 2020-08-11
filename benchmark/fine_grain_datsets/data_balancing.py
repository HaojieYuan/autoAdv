
import operator
import argparse
import random
from PIL import Image
import os
import hashlib

import pdb

parser = argparse.ArgumentParser(description='Balancing data for FGVC datasets.')
parser.add_argument('--prefix', default='', type=str)
parser.add_argument('--list', default='', type=str)
parser.add_argument('--outprefix', default='', type=str)
args = parser.parse_args()

# count for each class

class_counts = {}
class_imgs = {}

f = open(args.list)
for line in f:
    img_name, class_id = line.strip().split(' ')
    class_id = int(class_id)
    if class_id not in class_counts.keys():
        class_counts[class_id] = 1
        class_imgs[class_id] = [img_name]
    else:
        class_counts[class_id] += 1
        class_imgs[class_id].append(img_name)

f.close()




# get max value

max_class_id = max(class_counts.items(), key=operator.itemgetter(1))[0]
max_num = class_counts[max_class_id]

pdb.set_trace()


# pad class with small img amount
filters= ['gaussian', 'mode', 'median', 'max', 'min']
for class_id in class_counts.keys():
    this_class_num = class_counts[class_id]
    for i in range(max_num-this_class_num):
        img_to_aug = random.choice(class_imgs[class_id])
        img = Image.open(os.peht.join(args.prefix, img_to_aug))
        transform= random.choice(filters)
        if transform== 'gaussian':
            new_img= img.filter(ImageFilter.GaussianBlur(radius=3))
        elif transform== 'mode':
            new_img= img.filter(ImageFilter.ModeFilter(size=9))
        elif transform== 'median':
            new_img= img.filter(ImageFilter.MedianFilter(size=9))
        elif transform=='max':
            new_img= img.filter(ImageFilter.MaxFilter(size=9))
        else:
            new_img= img.filter(ImageFilter.MinFilter(size=9))

        new_img_id = (hashlib.md5(new_img).tobytes()).hexdigest()
        new_img.save(os.path.join(args.outprefix, new_img_id+'.png'), 'PNG')







