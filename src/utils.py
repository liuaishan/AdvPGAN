import PIL
import numpy as np
import tensorflow as tf
import pickle
import os
import matplotlib.pyplot as plt


# todo read label

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

