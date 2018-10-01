# Image-Style-Transform
In our project, we use the perceptual loss functions for training feed-forward networks instead of the traditional per-pixel loss between the output and ground-truth images (style image and content image) for image style transformation. The method we use can do the transformation almost in real- time via GPU. We divide our neural network into two parts, the image transform net and the loss net. For the image transform net we compare the result of two image transform net, with or without residual blocks. By the method of extracting high-level features from pre-trained loss net, we can optimize the quality of images generated. Our loss function contains three different parts, that is, style reconstruction loss, feature reconstruction loss and total variation loss for style penalty, content penalty and spatial smoothness in the output image separately. Also, we do a multiple style transfer and make some improvements for the quality of the transformed image.

How to use?\n
simply run python3 train.py and transform.py\n

For changing parameters:\n
In train.py:\n
    style_weight= 4.0\n
    content_weight= 1.0\n
    tv_weight= 1e-6\n
    learning_rate = 1e-3\n
    style= 'la_muse'\n
    train_image_path = "images/train/"\n
    style_image_path = "images/style/"+style+".jpg"\n
    model.save_weights(style+'_weights2.h5')\n
In transform.py:\n
    style= 'wave_crop'\n
    input_file = 'images/content/8.jpeg'\n
    model.load_weights(style+'_weights.h5',by_name=False)\n
    imsave('res8.png', y)\n
