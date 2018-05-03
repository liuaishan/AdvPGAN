import PIL
import numpy as np
import tensorflow as tf
import pickle
import os
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import OneHotEncoder
import cv2

# change list of labels to one hot encoder
# e.g. [0,1,2] --> [[1,0,0],[0,1,0],[0,0,1]]
def OHE_labels(Y_tr, N_classes):
    OHC = OneHotEncoder()
    Y_ohc = OHC.fit(np.arange(N_classes).reshape(-1, 1))
    Y_labels = Y_ohc.transform(Y_tr.reshape(-1, 1)).toarray()
    return Y_labels

# apply histogram equalization to remove the effect of brightness, use openCV'2 cv2
# scale images between -.5 and .5, by dividing by 255. and subtracting .5.
# this function can be merged with preprocess_image(img, image_size)
def pre_process_image(image):
    image[:,:,0] = cv2.equalizeHist(image[:,:,0])
    image[:,:,1] = cv2.equalizeHist(image[:,:,1])
    image[:,:,2] = cv2.equalizeHist(image[:,:,2])
    image = image/255.-.5
    return image


# do rotation, translation and shear in the image
def transform_image(image,ang_range,shear_range,trans_range):
    # Rotation
    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = image.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])
    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2
    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
    shear_M = cv2.getAffineTransform(pts1,pts2)
        
    image = cv2.warpAffine(image,Rot_M,(cols,rows))
    image = cv2.warpAffine(image,Trans_M,(cols,rows))
    image = cv2.warpAffine(image,shear_M,(cols,rows))
    
    image = pre_process_image(image)
    
    return image


# generate extra data
def gen_extra_data(X_train,y_train,N_classes,n_each,ang_range,shear_range,trans_range,randomize_Var): 
    n_class = len(np.unique(y_train)) 
    X_arr = []
    Y_arr = []
    n_train = len(X_train)
    for i in range(n_train):
        for i_n in range(n_each):
            img_trf = transform_image(X_train[i],
                                      ang_range,shear_range,trans_range)
            X_arr.append(img_trf)
            Y_arr.append(y_train[i])
            
    X_arr = np.array(X_arr,dtype = np.float32())
    Y_arr = np.array(Y_arr,dtype = np.float32())
    
    if (randomize_Var == 1):
        len_arr = np.arange(len(Y_arr))
        np.random.shuffle(len_arr)
        X_arr[len_arr] = X_arr
        Y_arr[len_arr] = Y_arr

    # liuaishan 2018.5.3 N_classes should be used instead of a constant
    labels_arr = OHE_labels(Y_arr,N_classes)

    return X_arr,Y_arr,labels_arr

# ZhangAnlan 2018.5.3
# param@num number of image/patch to load
# param@data_dir directory of image/patch
# returnVal@ return a pair of list of image/patch and corresponding labels i.e return image, label
# extra=True --> need to generate extra data, otherwise only preprocess
# N_classes, n_each=, ang_range, shear_range, trans_range and randomize_Var are parameters needed to generate extra data
def load_image( num, data_dir, N_classes, encode='latin1' , extra=False, n_each=5, ang_range=10, shear_range=2, trans_range=2, randomize_Var=1):
    image = []
    label = []
    while(len(label) < num):
        with open(data_dir, 'rb') as f:
            # cifar-10 need use 'latin1'
            data = pickle.load(f, encoding=encode)
        # the names of the keys should be unified as 'data', 'labels'
        image = image + data['data']
        label = label + data['labels']
    if(extra):
        image, label, ohc_label = gen_extra_data(image[0:num], label[0:num], N_classes, n_each, ang_range, shear_range, trans_range, randomize_Var)
    else:
        image_temp = np.array([pre_process_image(image[i]) for i in range(len(image))], dtype=np.float32)
        image = image_temp
        label = label[0:num]
    return image, label

# load and augment patch, image with different combinations
def shuffle_augment_and_load(image_num, image_dir, patch_num, patch_dir, batch_size):

    if batch_size <= 0:
        return None

    # load image/patch from directory
    image_set, image_label_set = load_image(image_num, image_dir, N_classes=43)
    patch_set = load_image(patch_num, patch_dir, N_classes=10)

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

