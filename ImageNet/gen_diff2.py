from __future__ import print_function
import numpy as np
import os
import PIL
import tensorflow as tf
import random
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.python.framework.ops import enable_eager_execution

disable_eager_execution()

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
import argparse
from tensorflow.keras.layers import Input
import imageio

from configs import bcolors
from utils import *
import tensorflow_model_optimization as tfmot


from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19

import re
def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f)]

# input image dimensions
img_rows, img_cols = 224, 224
input_shape = (img_rows, img_cols, 3)

K.clear_session()
# define input tensor as a placeholder
input_tensor = Input(shape=[img_rows, img_cols, 3],dtype=tf.float32 )

# load multiple models sharing same input tensor
K.set_learning_phase(0)
model1 = VGG16(input_tensor=input_tensor)

tf.keras.mixed_precision.set_global_policy('mixed_float16')
model2 = VGG16(input_tensor=input_tensor)

transformation = 'light'
start_point = (0, 0)
occlusion_size =(50, 50)
step = 10
grad_iterations = 100

img_paths = list_pictures('./seeds/', ext='JPEG')
num_img = 0
same_lable = 0
for img_path in img_paths:
    gen_img = preprocess_image(img_path)
    orig_img = gen_img.copy()
    # first check if input already induces differences
    pred1, pred2 = model1.predict(gen_img), model2.predict(gen_img)
    label1, label2 = np.argmax(pred1[0]), np.argmax(pred2[0])
    
    if not decode_label(pred1) == decode_label(pred2):
        gen_img_deprocessed = deprocess_image(gen_img)
        # save the result to disk
        imageio.imwrite('./new_generated_inputs/' + 'already_differ_' + decode_label(pred1) + '_' + decode_label(
            pred2) + '.png', gen_img_deprocessed)
        continue

    same_lable += 1
    # construct joint loss function
    orig_label = label1
    loss1 = K.mean(model1.get_layer('predictions').output[..., orig_label])
    loss2 = K.mean(model2.get_layer('predictions').output[..., orig_label])
    layer_output = tf.dtypes.cast(loss1,tf.float16) - loss2
    
    # for adversarial image generation
    final_loss = K.mean(layer_output)

    # we compute the gradient of the input picture wrt this loss
    grads = normalize(K.gradients(final_loss, input_tensor)[0])

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_tensor], [loss1, loss2, grads])
    
    for iters in range(0,grad_iterations):
        loss_value1, loss_value2, grads_value = iterate(gen_img)
        if transformation == 'light':
            grads_value = constraint_light(grads_value)  # constraint the gradients value
        elif transformation == 'occl':
            grads_value = constraint_occl(grads_value, start_point,occlusion_size)  # constraint the gradients value
        elif transformation == 'blackout':
            grads_value = constraint_black(grads_value)  # constraint the gradients value

        gen_img += grads_value * step
        pred1, pred2= model1.predict(gen_img), model2.predict(gen_img)
        label1, label2 = np.argmax(pred1[0]), np.argmax(pred2[0])
        if not decode_label(pred1) == decode_label(pred2):
            num_img += 1
            gen_img_deprocessed = deprocess_image(gen_img)
            orig_img_deprocessed = deprocess_image(orig_img)
            # save the result to disk
            imageio.imwrite(
                './new_generated_inputs/' + transformation + '_' + decode_label(pred1) + '_' + decode_label(
                    pred2) + '.png', gen_img_deprocessed)
            imageio.imwrite(
                './new_generated_inputs/' + transformation + '_' + decode_label(pred1) + '_' + decode_label(
                    pred2) + '_orig.png', orig_img_deprocessed)
            break

print('num of images with same label:   ' + same_lable)
print('num of ad examples generated:    ' + num_img)