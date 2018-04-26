import PIL
import numpy as np
import tensorflow as tf
import pickle
import os
import matplotlib.pyplot as plt
import random


# todo
# param@num number of image/patch to load
# param@data_dir directory of image/patch
# returnVal@ return a pair of list of image/patch and corresponding labels i.e return image, label
def load_image(num, data_dir, encode):
    image = []
    label = []
    while(len(label) < num):
        with open(data_dir, 'rb') as f:
            # cifar-10 need use 'latin1'
            data = pickle.load(f, encoding=encode)
        # the names of the keys should be unified as 'data', 'labels'
        image = image + data['data']
        label = label + data['labels']
    return image[0:num], label[0:num]

# load and augment patch, image with different combinations
def shuffle_augment_and_load(image_num, image_dir, patch_num, patch_dir, batch_size):

    if batch_size <= 0:
        return None

    # load image/patch from
    image_set, image_label_set = load_image(image_num, image_dir)
    patch_set = load_image(patch_num, patch_dir)

    result_img = []
    result_patch = []
    result_img_label = []

    # all combinations for images and patches
    for i in range(image_num):
        for j in range(patch_num):
            result_img.append(image_set[i])
            result_img_label.append(image_label_set[i])
            result_patch.append(patch_set[j])

    # more, need to be croped
    if len(result_img) >= batch_size:
        result_img = result_img[:batch_size]
        result_img_label = result_img_label[:batch_size]
        result_patch = result_patch[:batch_size]
        return result_img,result_img_label,result_patch

    # not enough, random pick
    else:
        len = batch_size - len(result_img)
        for iter in range(len):
            result_img.append(image_set[random.randint(0, image_num - 1)])
            result_img_label.append(image_label_set[random.randint(0, image_num - 1)])
            result_patch.append(patch_set[random.randint(0, patch_num - 1)])
        return result_img,result_img_label,result_patch


# preprocess the image
def preprocess_image(img, image_size=128):
    big_dim = max(img.width, img.height)
    wide = img.width > img.height
    new_w = image_size if not wide else int(img.width * image_size / img.height)
    new_h = image_size if wide else int(img.height * image_size / img.width)
    img = img.resize((new_w, new_h)).crop((0, 0, image_size, image_size))
    img = (np.asarray(img) / 255.0).astype(np.float32)
    return img

# load image and corresponding label
def load_data(img_path, image_size=299):
    img = PIL.Image.open(img_path)

    # liuaishan 2018.4.12 for python2.7, remove later
    # img.width = img.size[0]
    # img.height = img.size[1]

    big_dim = max(img.width, img.height)
    wide = img.width > img.height
    new_w = image_size if not wide else int(img.width * image_size / img.height)
    new_h = image_size if wide else int(img.height * image_size / img.width)
    img = img.resize((new_w, new_h)).crop((0, 0, image_size, image_size))
    img = (np.asarray(img) / 255.0).astype(np.float32)

    # todo read label
    y_hat=10
    label = tf.one_hot(y_hat, 1000)
    return img, label

# save tensor
def save_obj(tensor, filename):
    tensor = np.asarray(tensor).astype(np.float32)
    # print(b.eval())
    serialized = pickle.dumps(tensor, protocol=0)
    with open(filename, 'wb') as f:
        f.write(serialized)

# load tensor
def load_obj(filename):
    if os.path.exists(filename):
        return None
    with open(filename, 'rb') as f:
        tensor = pickle.load(f)
    tensor = np.asarray(tensor).astype(np.float32)
    return tensor

def _convert(image):
    return (image * 255.0).astype(np.uint8)

# show image
def show_image(image):
    plt.axis('off')
    plt.imshow(_convert(image), interpolation="nearest")
    plt.show()

