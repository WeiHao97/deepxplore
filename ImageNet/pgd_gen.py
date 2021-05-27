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
import tensorflow_datasets as tfds
import argparse
from tensorflow.keras.layers import Input
import scipy.misc

from configs import bcolors
from utils import *
import tensorflow_model_optimization as tfmot
from tensorflow.keras.applications.resnet50 import ResNet50

import time


BATCH_SIZE = 50
c = 1
grad_iterations = 20
step = 1
epsilon = 8

es = {'file_name': tf.TensorSpec(shape=(), dtype=tf.string, name=None),
 'image': tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32, name=None),
 'label': tf.TensorSpec(shape=(), dtype=tf.int64, name=None)}
mydataset = tf.data.experimental.load("/local/rcs/wei/Final3kImagePerClass/",es).batch(BATCH_SIZE).prefetch(1)

# input image dimensions

img_rows, img_cols = 224 ,224
model_ = ResNet50(input_shape= (img_rows, img_cols,3))
q_model = tfmot.quantization.keras.quantize_model(model_)
q_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model = ResNet50(input_tensor = q_model.input)
model.load_weights("./fp_model_40_resnet50.h5")

q_model.load_weights("./q_model_40_resnet50.h5")
preprocess = tf.keras.applications.resnet.preprocess_input

def second(image,label):
    input_image = image
    orig_logist = tf.identity(model(preprocess(input_image)[None,...]) )
    orig_label =  np.argmax(orig_logist[0])

    
    quant_logist = tf.identity(q_model(preprocess(input_image)[None,...]))
    quant_label =  np.argmax(quant_logist[0])

    
    if orig_label != quant_label:
        print(orig_label)
        return -2,-2,-2,-2,-2
    
    if orig_label != label:
        return -3,-3,-3,-3,-3
    
    A = 0
    start_time = time.time()
    for iters in range(0, grad_iterations):
        with tf.GradientTape() as g:
            g.watch(input_image)
            ad_img = preprocess(input_image + A)[None,...]
            final_loss = tf.keras.losses.categorical_crossentropy(orig_logist[0] , q_model(ad_img)[0])

        grads = normalize(g.gradient(final_loss, input_image))
        A += tf.sign(grads) * step
        A = tf.clip_by_value(A, -epsilon, epsilon)
        test_image_deprocess = tf.clip_by_value(input_image + A, 0, 255)
        test_image = preprocess(test_image_deprocess)[None,...]
        pred1, pred2= model(test_image), q_model(test_image)
        label1, label2 = np.argmax(pred1[0]), np.argmax(pred2[0])
        
        if not label1 == label2:
            if label1 == orig_label and tf.keras.applications.resnet.decode_predictions(pred1.numpy(), top=1)[0][0][2] > 0.6:
                

                total_time = time.time() - start_time
                
                gen_img_deprocessed = test_image_deprocess
                orig_img_deprocessed = input_image
                A = (gen_img_deprocessed - orig_img_deprocessed).numpy()
                
                norm = np.max(np.abs(A))
                
                return total_time, norm, iters, gen_img_deprocessed, A
    return -1, -1, -1, -1, -1

def topk(model_pred, qmodel_pred, k):
    preds = tf.keras.applications.resnet.decode_predictions(model_pred.numpy(), top=k)
    qpreds = tf.keras.applications.resnet.decode_predictions(qmodel_pred.numpy(), top=1)[0][0][1]
    
    for pred in preds[0]:
        if pred[1] == qpreds:
            return True
    
    return False

def secondk(image,k):
    input_image = image
    orig_img = tf.identity(input_image)
    orig_logist = tf.identity(model(preprocess(input_image)[None,...]))
    orig_label =  np.argmax(orig_logist[0])

    
    quant_logist = tf.identity(q_model(preprocess(input_image)[None,...]))
    quant_label =  np.argmax(quant_logist[0])

    
    if orig_label != quant_label:
        return -2,-2,-2,-2,-2
    
    A = 0
    start_time = time.time()
    for iters in range(0,grad_iterations):
        with tf.GradientTape() as g:
            g.watch(input_image)
            ad_img = preprocess(input_image + A)[None,...]
            final_loss = tf.keras.losses.categorical_crossentropy(orig_logist[0] , q_model(ad_img)[0])

        grads = normalize(g.gradient(final_loss, input_image))
        A += tf.sign(grads) * step
        A = tf.clip_by_value(A, -epsilon, epsilon)
        test_image_deprocess = tf.clip_by_value(input_image + A, 0, 255)
        test_image = preprocess(test_image_deprocess)[None,...]
        pred1, pred2= model(test_image), q_model(test_image)
        label1, label2 = np.argmax(pred1[0]), np.argmax(pred2[0])
        
        if not topk(pred1, pred2, k):
            if label1 == orig_label and tf.keras.applications.resnet.decode_predictions(pred1.numpy(), top=1)[0][0][2] > 0.6:
        
                total_time = time.time() - start_time
                
                gen_img_deprocessed = test_image_deprocess
                orig_img_deprocessed = input_image
                A = (gen_img_deprocessed - orig_img_deprocessed).numpy()
                norm = np.max(np.abs(A))
                
                return total_time, norm, iters, gen_img_deprocessed, A
            
    return -1, -1, -1, -1, -1

def calc_normal_success(method, methodk, ds, folderName='', filterName='',dataName='',dataFolder='',locald = ''):
    
    total=0
    count=0
    badimg = 0
    
    top5 = 0

    timeStore = []
    advdistStore = []
    stepsStore = []
    
    timeStorek = []
    advdistStorek = []
    stepsStorek = []
    
    for i, features in enumerate(ds):

        images = features['image']
        labels = features['label']

        for j,image in enumerate(images):
            
            label = labels[j].numpy()

            time, advdist, steps, gen, A = method(image,label)

            total += 1

            if time == -1:
                print("Didnt find anything")
                continue
            
            if time == -2:
                badimg += 1
                total -= 1
                print("Bad Image",badimg)
                continue
                
            if time == -3:
                badimg += 1
                total -= 1
                print("Incorrect Image",badimg)
                continue

            count += 1
            np.save(locald+folderName+"/"+dataName+str(count)+"@"+str(total)+".npy", gen)
            np.save(locald+filterName+"/"+dataName+str(count)+"@"+str(total)+".npy", A)
            
            timeStore.append(time)
            advdistStore.append(advdist)
            stepsStore.append(steps)
            
            with open(locald+dataFolder+"/"+dataName+'_time_data.csv', 'a') as f:
                f.write(str(time) + ", ")

            with open(locald+dataFolder+"/"+dataName+'_advdist_data.csv', 'a') as f:
                f.write(str(advdist) + ", ")
            
            with open(locald+dataFolder+"/"+dataName+'_steps_data.csv', 'a') as f:
                f.write(str(steps) + ", ")
                
            print("starting k search")
            
            time, advdist, steps, gen, A = methodk(image,5)
            
            if time == -1:
                print("Didnt find anything in K")
                continue
            
            if time == -2:
                print("Bad Image in K",badimg)
                continue
            
            top5 += 1
            
            np.save(locald+folderName+"/"+dataName+"k"+str(count)+".npy", gen)
            np.save(locald+filterName+"/"+dataName+"k"+str(count)+".npy", A)
            
            timeStorek.append(time)
            advdistStorek.append(advdist)
            stepsStorek.append(steps)
        
            with open(locald+dataFolder+"/"+dataName+'_timek_data.csv', 'a') as f:
                f.write(str(time) + ", ")

            with open(locald+dataFolder+"/"+dataName+'_advdistk_data.csv', 'a') as f:
                f.write(str(advdist) + ", ")
            
            with open(locald+dataFolder+"/"+dataName+'_stepsk_data.csv', 'a') as f:
                f.write(str(steps) + ", ")

            print("Number seen:",total)
            print("No. worked:", count)
            print("No. topk:", top5)


calc_normal_success(second,secondk,mydataset,
                   folderName='resnet_imagenet_images_second', filterName='resnet_imagenet_filters_second',dataName='second', dataFolder='resnet_imagenet_data_second', locald ='/local/rcs/wei/PGD/resnet/' )