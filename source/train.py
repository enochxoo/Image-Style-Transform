
# coding: utf-8

# In[1]:


from loss import dummy_loss
from whole_net_res import image_transform_net, loss_net
from keras.layers import Input, merge
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


# In[2]:


style_weight= 4.0
content_weight= 1.0
tv_weight= 1e-6
learning_rate = 1e-3
style= 'la_muse'
img_width = img_height = 256

train_image_path = "images/train/"
style_image_path = "images/style/"+style+".jpg"

net = image_transform_net(img_width,img_height,tv_weight)
model = loss_net(net.output,net.input,img_width,img_height,style_image_path,content_weight,style_weight)
model.summary()

optimizer = Adam(lr = learning_rate) 
model.compile(optimizer,  dummy_loss)  # Dummy loss since we are learning from regularizes

datagen = ImageDataGenerator()
train_generator = datagen.flow_from_directory(train_image_path, class_mode='input', batch_size=1, target_size=(img_width, img_height), shuffle=True)

model.fit_generator(verbose=1, generator = train_generator, steps_per_epoch=80000, epochs=2)
model.save_weights(style+'_weights2.h5')

