import numpy as np
import os
import json
import time

os.environ["CUDA_VISIBLE_DEVICES"]="7"
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import tensorflow as tf
import re
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.python.framework.ops import enable_eager_execution
#disable_eager_execution()
enable_eager_execution()

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.layers import Input

from utils import *
import tensorflow_model_optimization as tfmot

# input image dimensions
img_rows, img_cols = 224 ,224
input_shape = (img_rows, img_cols, 3)
model_ = ResNet50(input_shape=input_shape)
q_model = tfmot.quantization.keras.quantize_model(model_)
model = ResNet50(input_tensor = q_model.input)
model.load_weights("./original_model_40.h5")
q_model.load_weights("./int4_model_40.h5")


images = np.load('/local/rcs/wei/image_net/one_im_per_class.npy')
c = 1e2
grad_iterations = 20
step = 1
epsilon = 8

As = {}
int4_labels = []
start_time = time.time()
for image in images:
    input_image = tf.convert_to_tensor(image)
    orig_logist = tf.identity(model(image))
    orig_label =  np.argmax(orig_logist[0])
    A = 0
    success = False
    for iters in range(0,grad_iterations):
        with tf.GradientTape() as g:
            g.watch(input_image)
            loss1 = tf.keras.losses.MSE(model(input_image + A)[0] , q_model(input_image + A)[0])
            #increases this loss, c should be large to make this more importatnt
            loss2 = tf.keras.losses.MSE(orig_logist[0] , model(input_image + A)[0])
            # decreases this loss
            final_loss = loss1 - c*loss2
            #print(final_loss)


            grads = normalize(g.gradient(final_loss, input_image))
        A += tf.sign(grads) * step
        A = tf.clip_by_value(A, -epsilon, epsilon)
        test_image_deprocess = deprocess_image((input_image + A).numpy())
        test_image = np.expand_dims(tf.keras.applications.resnet.preprocess_input(test_image_deprocess), axis=0)
        pred1, pred2= model(test_image), q_model(test_image)
        label1, label2 = np.argmax(pred1[0]), np.argmax(pred2[0])
        l_1 = tf.keras.applications.resnet.decode_predictions(pred1.numpy(), top=5)[0][0][1]
        l_2 = tf.keras.applications.resnet.decode_predictions(pred2.numpy(), top=5)[0][0][1]
        if not l_1 == l_2:
            if label1 == orig_label and tf.keras.applications.resnet.decode_predictions(pred1.numpy(), top=5)[0][0][2] > 0.6:
                int4_labels.append(orig_label)
                np.save('/local/rcs/wei/image_net/one_image_per_class/int4/' + str(orig_label)+ '_' + l_1 +'_' + l_2+ '.npy',test_image_deprocess)
                success = True
                break
    if success:
        success = False
        As[int(orig_label)] = A.numpy().tolist()

json = json.dumps(As)
f = open("/local/rcs/wei/image_net/one_image_per_class/int4/attacks.json","w")
f.write(json)
f.close()
np.save("/local/rcs/wei/image_net/one_image_per_class/int4/labels.npy",np.array(int4_labels))
print("--- %s seconds ---" % (time.time() - start_time))
