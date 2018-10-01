
# coding: utf-8

# In[1]:


from loss import dummy_loss
from keras.optimizers import Adam
from scipy.misc import imsave
import numpy as np 
import h5py
import tensorflow as tf
from whole_net_res import image_transform_net, loss_net
from scipy.ndimage.filters import median_filter
from preprocess import preprocess_reflect_image, crop_image


# In[2]:


def median_filter_all_colours(im_small, window_size):
    """
    Applies a median filer to all colour channels
    """
    ims = []
    for d in range(3):
        im_conv_d = median_filter(im_small[:,:,d], size=(window_size,window_size))
        ims.append(im_conv_d)

    im_conv = np.stack(ims, axis=2).astype("uint8")
    
    return im_conv


# In[3]:


style= 'wave_crop'
input_file = 'images/content/8.jpeg'

aspect_ratio, x = preprocess_reflect_image(input_file)
img_width= img_height = x.shape[1]

net = image_transform_net(img_width,img_height)
model = loss_net(net.output,net.input,img_width,img_height,"",0,0)
model.compile(Adam(),  dummy_loss)  # Dummy loss since we are learning from regularizes
model.load_weights(style+'_weights.h5',by_name=False)

y = net.predict(x)[0] 
y = crop_image(y, aspect_ratio)
y = median_filter_all_colours(y, 3)

imsave('res8.png', y)

