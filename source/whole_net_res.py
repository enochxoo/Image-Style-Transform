#from loss import content_loss, style_loss, total_variation_loss, dummy_loss
from layers import InputNormalize,VGGNormalize,ReflectionPadding2D,Denormalize,conv_bn_relu,res_conv,dconv_bn_nolinear
from keras.layers.merge import concatenate, add
from keras.layers import Input
from keras import backend as K
from VGG19 import VGG19
from VGG16 import VGG16
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Conv2D, Conv2DTranspose
from keras.layers.convolutional import Conv2DTranspose, Cropping2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Model,Sequential
from keras.layers.normalization import BatchNormalization
from keras.engine.topology import Layer
from preprocess import preprocess_image
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


#https://keunwoochoi.wordpress.com/2016/03/09/residual-networks-implementation-on-keras/
def res_conv(nb_filter, nb_row, nb_col,stride=(1,1)):
    def _res_func(x):
        identity = Cropping2D(cropping=((2,2),(2,2)))(x)

        a = Conv2D(nb_filter, (nb_row, nb_col), strides=stride, padding='valid')(x)
        a = BatchNormalization()(a)
        #a = LeakyReLU(0.2)(a)
        a = Activation("relu")(a)
        a = Conv2D(nb_filter, (nb_row, nb_col), strides=stride, padding='valid')(a)
        y = BatchNormalization()(a)

        return  add([identity, y])

    return _res_func


def image_transform_net(img_width, img_height, tv_weight=1):
    
    x = Input(shape=(img_width,img_height,3), name='itn_input')
    a = InputNormalize(name='itn_input_norm')(x)
    a = ReflectionPadding2D(padding=(40,40),input_shape=(img_width,img_height,3), name='itn_reflectpad')(a)
    
    a = Conv2D(32, (9,9), strides=1, padding='same', name='conv_1')(a)
    a = BatchNormalization(name='batch_norm_1')(a)
    a = Activation('relu',name='act_1')(a)
    
    a = Conv2D(64, (3,3), strides=2, padding='same', name='conv_2')(a)
    a = BatchNormalization(name='batch_norm_2')(a)
    a = Activation('relu',name='act_2')(a)
    
    a = Conv2D(128, (3,3), strides=2, padding='same', name='conv_3')(a)
    a = BatchNormalization(name='batch_norm_3')(a)
    a = Activation('relu',name='act_3')(a)
    
    # Residual 1
    a = res_conv(128,3,3)(a)
    
    # Residual 2
    a = res_conv(128,3,3)(a)

    # Residual 3
    a = res_conv(128,3,3)(a)

    # Residual 4
    a = res_conv(128,3,3)(a)

    # Residual 5
    a = res_conv(128,3,3)(a)
        
    a = Conv2DTranspose(64,(3,3),strides=2,padding='same', name='conv_4')(a)
    a = BatchNormalization(name='batch_norm_4')(a)
    a = Activation('relu',name='act_4')(a)

    a = Conv2DTranspose(32,(3,3),strides=2,padding='same', name='conv_5')(a)
    a = BatchNormalization(name='batch_norm_5')(a)
    a = Activation('relu',name='act_5')(a)
    
    a = Conv2D(3, (9,9), strides=1, padding='same', name='conv_6')(a)
    a = BatchNormalization(name='batch_norm_6')(a)
    a = Activation('tanh',name='act_6')(a)    #output_image
    # Scale output to range [0, 255] via custom Denormalize layer
    y_hat = Scale_tanh(name='transform_output')(a)
    
    itn_model = Model(inputs=x, outputs=y_hat)
    #itn_model.load_weights('wave_crop_weights.h5', by_name=True)
    #print(model.output.shape)
    add_total_variation_loss(itn_model.layers[-1],tv_weight)
    return itn_model


# def assessment_net(itn_output):

#     base_model = MobileNet(alpha=1, include_top=False, pooling='avg', weights=None, input_tensor=itn_output)
#     x = Dropout(0.75)(base_model.output)
#     x = Dense(10, activation='softmax')(x)

#     model = Model(base_model.input, x)
#     model.load_weights('mobilenet_weights.h5', by_name=True)
    
#     assess_loss = AssessmentRegularizer(ass_weight)(whole_net[-1])
#     assessment_net[-1].add_loss(assess_loss)

#     for layer in model.layers:
#         layer.trainable = False
    
#     return model

# class AssessmentRegularizer(Regularizer):
#     """ Johnson et al 2015 https://arxiv.org/abs/1603.08155 """

#     def __init__(self, weight=1.0):
#         self.weight = weight
#         super(AssessmentRegularizer, self).__init__()

#     def __call__(self, x):
#         scores = x.output[0]
#         loss = - self.weight * K.mean(scores) # Generated by network
#         return loss

def loss_net(x_in, trux_x_in,width, height,style_image_path,content_weight,style_weight):
    # Append the initial input to the FastNet input to the VGG inputs
    x = concatenate([x_in, trux_x_in], axis=0)
    
    # Normalize the inputs via custom VGG Normalization layer
    x = VGGNormalize(name="vgg_normalize")(x)

    vgg = VGG16(include_top=False,input_tensor=x)

    vgg_output_dict = dict([(layer.name, layer.output) for layer in vgg.layers[-14:]])
    vgg_layers = dict([(layer.name, layer) for layer in vgg.layers[-14:]])

    if style_weight > 0:
        add_style_loss(vgg,style_image_path , vgg_layers, vgg_output_dict, width, height,style_weight)   

    if content_weight > 0:
        add_content_loss(vgg_layers,vgg_output_dict,content_weight)

#     # Freeze all VGG layers
#     for layer in vgg.layers[-19:]:
#         layer.trainable = False

    return vgg

def add_style_loss(vgg,style_image_path,vgg_layers,vgg_output_dict,img_width, img_height,weight):
    style_img = preprocess_image(style_image_path, img_width, img_height)
    print('Getting style features from VGG network.')

    style_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']

    style_layer_outputs = []

    for layer in style_layers:
        style_layer_outputs.append(vgg_output_dict[layer])

    vgg_style_func = K.function([vgg.layers[-15].input], style_layer_outputs)

    style_features = vgg_style_func([style_img])

    # Style Reconstruction Loss
    for i, layer_name in enumerate(style_layers):
        layer = vgg_layers[layer_name]

        feature_var = K.variable(value=style_features[i][0])
        style_loss = StyleReconstructionRegularizer(
                            style_feature_target=feature_var,
                            weight=weight)(layer)

        layer.add_loss(style_loss)

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