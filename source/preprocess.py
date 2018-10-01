from scipy.misc import imread, imresize, imsave, fromimage, toimage
import numpy as np
import os
import tensorflow as tf
from keras import backend as K
from keras.preprocessing import image

def preprocess_image(path, width=256, height=256):
    img = imread(path, mode="RGB")
    img = imresize(img, (width, height),interp='nearest')
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

def preprocess_reflect_image(image_path):
    img = imread(image_path, mode="RGB")  # Prevents crashes due to PNG images (ARGB)
    org_w = img.shape[0]
    org_h = img.shape[1]
    
    aspect_ratio = org_h/org_w
    sw = (org_w // 4) * 4 # Make sure width is a multiple of 4
    sh = (org_h // 4) * 4 # Make sure width is a multiple of 4

    size  = max(sw, sh)
    pad_w = (size - sw) // 2
    pad_h = (size - sh) // 2

    tf_session = K.get_session()
    kvar = K.variable(value=img)

    paddings = [[pad_w,pad_w],[pad_h,pad_h],[0,0]]
    squared_img = tf.pad(kvar,paddings, mode='REFLECT', name=None)
    
    img = K.eval(squared_img)
    img = imresize(img, (size, size),interp='nearest')
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    
    return (aspect_ratio  ,img)


def crop_image(img, aspect_ratio):
    if aspect_ratio >1:
        w = img.shape[0]
        h = int(w // aspect_ratio)
        img =  K.eval(tf.image.crop_to_bounding_box(img, (w-h)//2,0,h,w))
    else:
        h = img.shape[1]
        w = int(h // aspect_ratio)
        img = K.eval(tf.image.crop_to_bounding_box(img, 0,(h-w)//2,h,w))
    return img