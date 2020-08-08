

# Oxford Flowers
CUDA_VISIBLE_DEVICES=8 python dataset2tfrecords.py \
    --folder "/home/haojieyuan/Data/OxfordFlowers/jpg" \
    --listfile "/home/haojieyuan/Data/OxfordFlowers/trainval_list.txt" \
    --out "/home/haojieyuan/Data/OxfordFlowers/train.tfrecords"


CUDA_VISIBLE_DEVICES=8 python dataset2tfrecords.py \
    --folder "/home/haojieyuan/Data/OxfordFlowers/jpg" \
    --listfile "/home/haojieyuan/Data/OxfordFlowers/test_list.txt" \
    --out "/home/haojieyuan/Data/OxfordFlowers/test.tfrecords"


# FGVC Aircraft
CUDA_VISIBLE_DEVICES=8 python dataset2tfrecords.py \
    --folder "/home/haojieyuan/Data/FGVC_Aircraft/fgvc-aircraft-2013b/data/images" \
    --listfile "/home/haojieyuan/Data/FGVC_Aircraft/fgvc-aircraft-2013b/data/trainval_list.txt" \
    --out "/home/haojieyuan/Data/FGVC_Aircraft/train.tfrecords"


CUDA_VISIBLE_DEVICES=8 python dataset2tfrecords.py \
    --folder "/home/haojieyuan/Data/FGVC_Aircraft/fgvc-aircraft-2013b/data/images" \
    --listfile "/home/haojieyuan/Data/FGVC_Aircraft/fgvc-aircraft-2013b/data/test_list.txt" \
    --out "/home/haojieyuan/Data/FGVC_Aircraft/test.tfrecords"


# Stanford Cars
CUDA_VISIBLE_DEVICES=8 python dataset2tfrecords.py \
    --folder "/home/haojieyuan/Data/stanfordCars/cars_train" \
    --listfile "/home/haojieyuan/Data/stanfordCars/train_list.txt" \
    --out "/home/haojieyuan/Data/stanfordCars/train.tfrecords"


CUDA_VISIBLE_DEVICES=8 python dataset2tfrecords.py \
    --folder "/home/haojieyuan/Data/stanfordCars/cars_test" \
    --listfile "/home/haojieyuan/Data/stanfordCars/test_list.txt" \
    --out "/home/haojieyuan/Data/stanfordCars/test.tfrecords"