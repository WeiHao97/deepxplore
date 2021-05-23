import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import PIL
import tensorflow as tf
import random
import re
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.python.framework.ops import enable_eager_execution
#disable_eager_execution()
enable_eager_execution()
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow_datasets as tfds
import argparse
from tensorflow.keras.layers import Input
import scipy.misc

from configs import bcolors
from utils import *
import tensorflow_model_optimization as tfmot


BATCH_SIZE = 50

def gen_one_image(image, label, loss_func):
    input_image = tf.convert_to_tensor(image)
    orig_img = tf.identity(input_image)
    orig_logist = tf.identity(model(image))
    orig_label =  np.argmax(orig_logist[0])
    
    A = 0
    
    for iters in range(0, grad_iterations):
        with tf.GradientTape() as g:
            g.watch(input_image)
            if loss_func == 'MSE':
                final_loss = tf.keras.losses.MSE(orig_logist[0] , q_model(input_image+A)[0])
            if loss_func == 'CCE':
                final_loss = tf.keras.losses.categorical_crossentropy(orig_logist[0] , q_model(input_image+A)[0])
            if loss_func == 'MEAN':
                final_loss = 1.0 - K.mean(q_model(input_image+A)[..., orig_label])

        grads = normalize(g.gradient(final_loss, input_image))
        A += tf.sign(grads) * step
        A = tf.clip_by_value(A, -epsilon, epsilon)
        test_image_deprocess = deprocess_image((input_image + A).numpy())
        test_image = np.expand_dims(tf.keras.applications.resnet.preprocess_input(test_image_deprocess), axis=0)
        pred1, pred2= model(test_image), q_model(test_image)
        label1, label2 = np.argmax(pred1[0]), np.argmax(pred2[0])
        if not label1 == label2:
            if label1 == orig_label and tf.keras.applications.resnet.decode_predictions(pred1.numpy(), top=5)[0][0][2] > 0.6:
                return (1, 0, A.numpy().astype(np.int8), pred1, pred2)
    
    indicator = 0 # 0: model right q_model wrong; 1: model right q_model wrong; 2: model wrong q_model right; 3: both wrong
    if label1 != orig_label and label2 != orig_label:
        indicator = 3
    elif label1 != orig_label and label2 == orig_label:
        indicator = 2
    elif label1 == orig_label and label2 == orig_label:
        indicator = 1
    else:
        indicator = 0
    return (0, indicator, A.numpy().astype(np.int8), pred1, pred2)
    

def get_n_images(n, loss_func, ds):
    d = {"both right": 0, "both wrong": 0, "model right q_model wrong": 0, "model wrong q_model right": 0}
    num = 0
    suc_cnt = 0
    for i, features in enumerate(ds):
        for j in range(BATCH_SIZE):
            if os.path.exists("/local/rcs/wei/attack_baseline/baseline_gen_" + loss_func + "/"+str(num)+"_filter.npy"):
                num += 1
                continue
            image = np.expand_dims(features['image'][j], axis=0)
            label = features['label'][j]
            pred1, pred2= model(image), q_model(image)
            label1, label2 = np.argmax(pred1[0]), np.argmax(pred2[0])
            if label1 != label2 or label1 != label:
                num += 1
                continue
            success, indicator, fil, pred1, pred2 = gen_one_image(image, label, loss_func)
            suc_cnt += success
            np.save("/local/rcs/wei/attack_baseline/baseline_gen_" + loss_func + "/"+str(num)+"_filter.npy", fil)
            if indicator == 0:
                d["model right q_model wrong"] += 1
            elif indicator == 1:
                d["both right"] += 1
            elif indicator == 2:
                d["model wrong q_model right"] += 1
            else:
                d["both wrong"] += 1
            num += 1
            if num == n:
                return suc_cnt, d
        print((i+1)*50)

    if (i + 1) % 50 == 0:
            print("Finished %d examples" % ((i + 1) * BATCH_SIZE))
            print(d)
    return suc_cnt, d


es = {'file_name': tf.TensorSpec(shape=(), dtype=tf.string, name=None),
 'image': tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32, name=None),
 'label': tf.TensorSpec(shape=(), dtype=tf.int64, name=None)}
mydataset = tf.data.experimental.load("/local/rcs/wei/Avg3ImagePerClass/",es).batch(BATCH_SIZE).prefetch(1)

# input image dimensions
img_rows, img_cols = 224 ,224
input_shape = (img_rows, img_cols, 3)
model_ = ResNet50(input_shape=input_shape)

q_model = tfmot.quantization.keras.quantize_model(model_)
model = ResNet50(input_tensor = q_model.input)
model.load_weights("./fp_model_40_resnet50.h5")
q_model.load_weights("./q_model_40_resnet50.h5")

grad_iterations = 20
step = 1
epsilon = 8

suc_cnt_cce, d_cce = get_n_images(3000, "CCE",mydataset)
print("loss function: CCE")
print("success rate: " + str(suc_cnt_cce/sum(d_cce.values())*100) + "%")
print(d_cce)