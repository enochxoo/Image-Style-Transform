from keras import backend as back

def dummy_loss(y_true, y_pred):
    return K.variable(0.0)

def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))

    shape = K.shape(x)
    
    C, W, H = (shape[0],shape[1], shape[2])
    
    cf = K.reshape(features ,(C,-1))
    gram = K.dot(cf, K.transpose(cf)) /  K.cast(C*W*H,dtype='float32')

    return gram

'''The content loss is a L2 distance between the features of the content 
image and the features of the generated image, keeping the generated image 
close enough to the original one.'''

def content_loss(content, generated，weight):
	return weight * K.sum(K.square(content - generated))

def style_loss(style, generated,weight):
	assert K.ndim(style) == 3
    assert K.ndim(generated) == 3
    S = gram_matrix(style)
    G = gram_matrix(generated)
    
    return weight * K.sum(K.square(S - G))

 def total_variation_loss(x,weight):
 	assert K.ndim(x) == 4
    shape = K.shape(x)
    image_width, image_height= (shape[1],shape[2])
    if K.image_data_format() == 'channels_first':
        a = K.square(x[:, :, :image_width - 1, :image_height - 1] - x[:, :, 1:, :image_height - 1])
        b = K.square(x[:, :, :image_width - 1, :image_height - 1] - x[:, :, :image_width - 1, 1:])
    else:
        a = K.square(x[:, :image_width - 1, :image_height - 1, :] - x[:, 1:, :image_height - 1, :])
        b = K.square(x[:, :image_width - 1, :image_height - 1, :] - x[:, :image_width - 1, 1:, :])
    
    return weight * K.sum(K.pow(a + b, 1.25))





