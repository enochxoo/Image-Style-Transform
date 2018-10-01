#from loss import content_loss, style_loss, total_variation_loss, dummy_loss
from keras.layers.merge import concatenate
from keras.layers import Input
from keras import backend as K
from VGG19 import VGG19
from VGG16 import VGG16
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Conv2D, Conv2DTranspose
from keras.layers.convolutional import Conv2DTranspose
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Model,Sequential
from keras.layers.normalization import BatchNormalization
from keras.engine.topology import Layer
from layers import InputNormalize,VGGNormalize,ReflectionPadding2D,Denormalize,conv_bn_relu,res_conv,dconv_bn_nolinear
import img_util
from keras.engine import InputSpec
import tensorflow as tf
from loss_old import StyleReconstructionRegularizer,FeatureReconstructionRegularizer,TVRegularizer
import numpy as np

class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), dim_ordering='default', **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)

        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()

        self.padding = padding
        if isinstance(padding, dict):
            if set(padding.keys()) <= {'top_pad', 'bottom_pad', 'left_pad', 'right_pad'}:
                self.top_pad = padding.get('top_pad', 0)
                self.bottom_pad = padding.get('bottom_pad', 0)
                self.left_pad = padding.get('left_pad', 0)
                self.right_pad = padding.get('right_pad', 0)
            else:
                raise ValueError('Unexpected key found in `padding` dictionary. '
                                 'Keys have to be in {"top_pad", "bottom_pad", '
                                 '"left_pad", "right_pad"}.'
                                 'Found: ' + str(padding.keys()))
        else:
            padding = tuple(padding)
            if len(padding) == 2:
                self.top_pad = padding[0]
                self.bottom_pad = padding[0]
                self.left_pad = padding[1]
                self.right_pad = padding[1]
            elif len(padding) == 4:
                self.top_pad = padding[0]
                self.bottom_pad = padding[1]
                self.left_pad = padding[2]
                self.right_pad = padding[3]
            else:
                raise TypeError('`padding` should be tuple of int '
                                'of length 2 or 4, or dict. '
                                'Found: ' + str(padding))

        if dim_ordering not in {'tf'}:
            raise ValueError('dim_ordering must be in {tf}.')
        self.dim_ordering = dim_ordering
        self.input_spec = [InputSpec(ndim=4)] 


    def call(self, x, mask=None):
        top_pad=self.top_pad
        bottom_pad=self.bottom_pad
        left_pad=self.left_pad
        right_pad=self.right_pad        
        
        paddings = [[0,0],[left_pad,right_pad],[top_pad,bottom_pad],[0,0]]

        
        return tf.pad(x,paddings, mode='REFLECT', name=None)

    def compute_output_shape(self,input_shape):
        if self.dim_ordering == 'tf':
            rows = input_shape[1] + self.top_pad + self.bottom_pad if input_shape[1] is not None else None
            cols = input_shape[2] + self.left_pad + self.right_pad if input_shape[2] is not None else None

            return (input_shape[0],
                    rows,
                    cols,
                    input_shape[3])
        else:
            raise ValueError('Invalid dim_ordering:', self.dim_ordering)
            
    def get_config(self):
        config = {'padding': self.padding}
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class InputNormalize(Layer):
    def __init__(self, **kwargs):
        super(InputNormalize, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        return x/255.
    
    def compute_output_shape(self,input_shape):
        return input_shape



class Scale_tanh(Layer):
    '''
    Custom layer to denormalize the final Convolution layer activations (tanh)
    Since tanh scales the output to the range (-1, 1), we add 1 to bring it to the
    range (0, 2). We then multiply it by 127.5 to scale the values to the range (0, 255)
    '''

    def __init__(self, **kwargs):
        super(Scale_tanh, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        '''
        Scales the tanh output activations from previous layer (-1, 1) to the
        range (0, 255)
        '''

        return (x + 1) * 127.5

    def compute_output_shape(self,input_shape):
        return input_shape



def image_transform_net(img_width, img_height, tv_weight=1):
    
    x = Input(shape=(img_width,img_height,3))
    a = InputNormalize()(x)
    a = ReflectionPadding2D(padding=(40,40),input_shape=(img_width,img_height,3))(a)
    
    a = Conv2D(32, (9,9), strides=1, padding='same')(a)
    a = BatchNormalization()(a)
    a = Activation('relu')(a)
    
    a = Conv2D(64, (3,3), strides=2, padding='same')(a)
    a = BatchNormalization()(a)
    a = Activation('relu')(a)
    
    a = Conv2D(128, (3,3), strides=2, padding='same')(a)
    a = BatchNormalization()(a)
    a = Activation('relu')(a)
    
    for i in range(5):
        a = Conv2D(128, (3,3), strides=1, padding='valid')(a)
        a = BatchNormalization()(a)
        a = Activation('relu')(a)
        a = Conv2D(128, (3,3), strides=1, padding='valid')(a)
        a = BatchNormalization()(a)
        a = Activation('relu')(a)
        
    a = Conv2DTranspose(64,(3,3),strides=2,padding='same')(a)
    a = BatchNormalization()(a)
    a = Activation('relu')(a)

    a = Conv2DTranspose(32,(3,3),strides=2,padding='same')(a)
    a = BatchNormalization()(a)
    a = Activation('relu')(a)
    
    a = Conv2D(3, (9,9), strides=1, padding='same')(a)
    a = BatchNormalization()(a)
    a = Activation('tanh')(a)    #output_image
    # Scale output to range [0, 255] via custom Denormalize layer
    y_hat = Scale_tanh(name='transform_output')(a)
    
    model = Model(inputs=x, outputs=y_hat)
    #print(model.output.shape)
    add_total_variation_loss(model.layers[-1],tv_weight)
    return model

def loss_net(x_in, trux_x_in,width, height,style_image_path, style_image_path2, content_weight,style_weight):
    # Append the initial input to the FastNet input to the VGG inputs
    x = concatenate([x_in, trux_x_in], axis=0)
    
    # Normalize the inputs via custom VGG Normalization layer
    x = VGGNormalize(name="vgg_normalize")(x)

    vgg = VGG16(include_top=False,input_tensor=x)

    vgg_output_dict = dict([(layer.name, layer.output) for layer in vgg.layers[-14:]])
    vgg_layers = dict([(layer.name, layer) for layer in vgg.layers[-14:]])

    if style_weight > 0:
        add_style_loss(vgg,style_image_path, style_image_path2, vgg_layers, vgg_output_dict, width, height,style_weight)   
    if content_weight > 0:
        add_content_loss(vgg_layers,vgg_output_dict,content_weight)

#     # Freeze all VGG layers
#     for layer in vgg.layers[-19:]:
#         layer.trainable = False

    return vgg

def add_style_loss(vgg,style_image_path, style_image_path2, vgg_layers,vgg_output_dict,img_width, img_height,weight):
    style_img = img_util.preprocess_image(style_image_path, img_width, img_height)
    style_img2 = img_util.preprocess_image(style_image_path2, img_width, img_height)
    print('Getting style features from VGG network.')

    style_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']

    style_layer_outputs = []

    for layer in style_layers:
        style_layer_outputs.append(vgg_output_dict[layer])

    vgg_style_func = K.function([vgg.layers[-15].input], style_layer_outputs)

    style_features = vgg_style_func([style_img])
    style_features2 = vgg_style_func([style_img2])

    # Style Reconstruction Loss
    for i, layer_name in enumerate(style_layers):
        layer = vgg_layers[layer_name]

        feature_var = K.variable(value=style_features[i][0])
        style_loss = StyleReconstructionRegularizer(
                            style_feature_target=feature_var,
                            weight=weight)(layer)
        feature_var2 = K.variable(value=style_features2[i][0])
        style_loss2 = StyleReconstructionRegularizer(
                            style_feature_target=feature_var2,
                            weight=weight)(layer)
        layer.add_loss(style_loss)
        layer.add_loss(style_loss2)

def add_content_loss(vgg_layers,vgg_output_dict,weight):
    # Feature Reconstruction Loss
    content_layer = 'block3_conv3'
    content_layer_output = vgg_output_dict[content_layer]

    layer = vgg_layers[content_layer]
    content_regularizer = FeatureReconstructionRegularizer(weight)(layer)
    layer.add_loss(content_regularizer)


def add_total_variation_loss(transform_output_layer,weight):
    # Total Variation Regularization
    layer = transform_output_layer  # Output layer
    tv_regularizer = TVRegularizer(weight)(layer)
    layer.add_loss(tv_regularizer)