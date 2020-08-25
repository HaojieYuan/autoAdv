
import os
from PIL import Image
import numpy as np

in_dir = '/home/haojieyuan/Data/caltech101/eval_imgs'
out_dir = '/home/haojieyuan/Data/caltech101/eval_imgs_resized'

#in_dir = '/home/haojieyuan/Data/oxfordFlowers/eval_imgs'
#out_dir = '/home/haojieyuan/Data/oxfordFlowers/eval_imgs_resized'

fraction = 0.875

for image_name in os.listdir(in_dir):
    target_height = 299.0
    target_width = 299.0

    im = Image.open(os.path.join(in_dir, image_name))
    im = im.convert('RGB')
    im_array = np.array(im)  # height, width, channel
    h, w ,c = im_array.shape
    h = float(h)
    w = float(w)

    box_h_start = int((h - h*fraction)/2)
    box_w_start = int((w - w*fraction)/2)

    h_size = int(h - 2*box_h_start)
    w_size = int(w - 2*box_w_start)

    croped = im_array[box_h_start:box_h_start+h_size, box_w_start:box_w_start+w_size, :]
    h, w, c = croped.shape
    h = float(h)
    w = float(w)

    if h < w:
        pad_h = True
    else:
        pad_h = False


    if pad_h:
        pad_upper = int((w-h)/2)
    else:
        pad_left = int((h-w)/2)

    pad_shape = int(max([h, w]))
    before_resize = np.zeros([pad_shape, pad_shape, 3])

    if pad_h:
        before_resize[pad_upper:pad_upper+h_size, :, :] = croped
    else:
        before_resize[:, pad_left:pad_left+w_size, :] = croped


    #ratio = max([h/target_height, w/target_width])
    #resized_h = h/ratio
    #resized_w = w/ratio
    #padding_h = (target_height-resized_h)/2.0
    #padding_w = (target_width -resized_w)/2.0




    im_out = Image.fromarray(np.uint8(before_resize))
    im_out = im_out.resize((int(target_height), int(target_width)))

    out_path = os.path.join(out_dir, image_name)
    im_out.save(out_path, 'PNG')



