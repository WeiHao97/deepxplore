import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"
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


class DefaultBNQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):

    def get_weights_and_quantizers(self, layer):
        return []

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        pass
    def set_quantize_activations(self, layer, quantize_activations):
        pass
    def get_output_quantizers(self, layer):
        return [tfmot.quantization.keras.quantizers.MovingAverageQuantizer(num_bits=8, per_axis=False, symmetric=False, narrow_range=False)]

    def get_config(self):
        return {}
    
    
class NoOpQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    """Use this config object if the layer has nothing to be quantized for 
    quantization aware training."""

    def get_weights_and_quantizers(self, layer):
        return []

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        pass

    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        # Does not quantize output, since we return an empty list.
        return []

    def get_config(self):
        return {}
    
    
def apply_quantization(layer):
    if 'bn'  in layer.name:
        return tfmot.quantization.keras.quantize_annotate_layer(layer,DefaultBNQuantizeConfig())
    elif 'concat' in layer.name:
        return tfmot.quantization.keras.quantize_annotate_layer(layer,NoOpQuantizeConfig())
    else:
        return tfmot.quantization.keras.quantize_annotate_layer(layer)

BATCH_SIZE = 50
c = 1
grad_iterations = 20
step = 1
epsilon = 8
mode = 'r'

es = {'file_name': tf.TensorSpec(shape=(), dtype=tf.string, name=None),
 'image': tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32, name=None),
 'label': tf.TensorSpec(shape=(), dtype=tf.int64, name=None)}
mydataset = tf.data.experimental.load("/local/rcs/wei/End3kImagePerClass/",es).batch(BATCH_SIZE).prefetch(1)

# input image dimensions

img_rows, img_cols = 224 ,224

if mode == 'm':
    model_ = tf.keras.applications.MobileNet(input_shape= (img_rows, img_cols,3))
    q_model = tfmot.quantization.keras.quantize_model(model_)
    model = tf.keras.applications.MobileNet(input_tensor = q_model.input)
    model.load_weights("./fp_model_40_mobilenet.h5")
    q_model.load_weights("./q_model_40_mobilenet.h5")
    model.trainable = False
    q_model.trainable = False
    preprocess = tf.keras.applications.mobilenet.preprocess_input
    decode = tf.keras.applications.mobilenet.decode_predictions
    net = 'mobilenet/'

elif mode == 'r':
    model_ = ResNet50(input_shape= (img_rows, img_cols,3))
    q_model = tfmot.quantization.keras.quantize_model(model_)
    model = ResNet50(input_shape= (img_rows, img_cols,3))
    d_model = ResNet50(input_tensor = q_model.input)
    model.load_weights("./fp_model_40_resnet50.h5")
    q_model.load_weights("./q_model_40_resnet50.h5")
    d_model.load_weights("./distilled_fp_model_40_resnet50.h5")
    model.trainable = False
    q_model.trainable = False
    d_model.trainable = False
    preprocess = tf.keras.applications.resnet.preprocess_input
    decode = tf.keras.applications.resnet.decode_predictions
    net = 'resnet-semibb/'

else:

    model_ = tf.keras.applications.DenseNet121(input_shape=(img_rows, img_cols,3))
    # Create a base model
    base_model = model_
    # Helper function uses `quantize_annotate_layer` to annotate that only the 
    # Dense layers should be quantized.

    LastValueQuantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer
    MovingAverageQuantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer
    
    # Use `tf.keras.models.clone_model` to apply `apply_quantization_to_dense` 
    # to the layers of the model.
    annotated_model = tf.keras.models.clone_model(
        base_model,
        clone_function=apply_quantization,
    )

    with tfmot.quantization.keras.quantize_scope({'DefaultBNQuantizeConfig': DefaultBNQuantizeConfig, 'NoOpQuantizeConfig': NoOpQuantizeConfig}):
        q_model = tfmot.quantization.keras.quantize_apply(annotated_model)

    model = tf.keras.applications.DenseNet121(input_tensor = q_model.input)
    model.load_weights("./fp_model_40_densenet121.h5")
    q_model.load_weights("./q_model_40_densenet121.h5")
    model.trainable = False
    q_model.trainable = False
    preprocess = tf.keras.applications.densenet.preprocess_input
    decode = tf.keras.applications.densenet.decode_predictions
    net = 'densenet/'


locald ='/local/rcs/wei/confidencedrop/' + net

def semi_bb(image,label):
        
    #ATTACK
    
    input_image = image
    
    img = np.copy(image)
    img = np.expand_dims(preprocess(img), axis=0)
    
    orig_img = tf.identity(input_image)
    orig_logist = tf.identity(model.predict(img))
    orig_label =  np.argmax(orig_logist[0])
    
    quant_logist = tf.identity(q_model.predict(img))
    quant_label =  np.argmax(quant_logist[0])

    d_logist =  tf.identity(d_model.predict(img))
    d_label =  np.argmax(d_logist[0])
    
    if orig_label != quant_label or orig_label != d_label:
        print(orig_label)
        return -2,-2,-2,-2,-2
    
    if orig_label != label:
        print(orig_label,label)
        return -3,-3,-3
    
    A = 0
    for iters in range(0,grad_iterations):
        with tf.GradientTape() as g:
            g.watch(input_image)
            loss1 = K.mean(d_model(preprocess(input_image + A)[None, ...], training=False)[..., orig_label])
            loss2 = K.mean(q_model(preprocess(input_image + A)[None, ...], training=False)[..., orig_label])
            final_loss = K.mean(loss1 - c*loss2)
       
        grads = normalize(g.gradient(final_loss, input_image))
        A += tf.sign(grads) * step
        A = tf.clip_by_value(A, -epsilon, epsilon)
        test_image_deprocess = tf.clip_by_value(input_image + A,0,255)
        test_image = np.expand_dims(preprocess(test_image_deprocess), axis=0)
        pred1, pred2= d_model.predict(test_image), q_model.predict(test_image)
        label1, label2 = np.argmax(pred1[0]), np.argmax(pred2[0])
        pred3 = model.predict(test_image)
        label3 = np.argmax(pred3[0])
        
        if not label1 == label2:
            if label1 == orig_label and decode(pred1, top=1)[0][0][2] > 0.6:

                gen_img_deprocessed = test_image_deprocess.numpy()
                orig_img_deprocessed = orig_img.numpy()
                
                A = gen_img_deprocessed - orig_img_deprocessed
        
                if label3 != orig_label:
                    return 5, -1, gen_img_deprocessed, A
                else:               
                    return 2, iters, gen_img_deprocessed, A
        
    gen_img_deprocessed = test_image_deprocess.numpy()
    orig_img_deprocessed = orig_img.numpy()

    A = gen_img_deprocessed - orig_img_deprocessed
        
        
    return 5, -1, gen_img_deprocessed, A

def pgd(image,label):
        
    #ATTACK
    
    input_image = image
    
    img = np.copy(image)
    img = np.expand_dims(preprocess(img), axis=0)
    
    orig_img = tf.identity(input_image)
    orig_logist = tf.identity(model.predict(img))
    orig_label =  np.argmax(orig_logist[0])
    
    quant_logist = tf.identity(q_model.predict(img))
    quant_label =  np.argmax(quant_logist[0])
    
    if orig_label != quant_label:
        return -2,-2,-2
    
    if orig_label != label:
        print(orig_label,label)
        return -3,-3,-3
    
    A = 0
    for iters in range(0,grad_iterations):
        with tf.GradientTape() as g:
            g.watch(input_image)
            ad_img = preprocess(input_image + A)[None,...]
            final_loss = tf.keras.losses.categorical_crossentropy(orig_logist[0] , q_model(ad_img, training = False)[0])
       
        grads = normalize(g.gradient(final_loss, input_image))
        A += tf.sign(grads) * step
        A = tf.clip_by_value(A, -epsilon, epsilon)
        test_image_deprocess = tf.clip_by_value(input_image + A,0,255)
        test_image = np.expand_dims(preprocess(test_image_deprocess), axis=0)
        pred1, pred2= model.predict(test_image), q_model.predict(test_image)
        label1, label2 = np.argmax(pred1[0]), np.argmax(pred2[0])
        
        if not label1 == label2:
            if label1 == orig_label and decode(pred1, top=1)[0][0][2] > 0.6:
                
                gen_img_deprocessed = test_image_deprocess.numpy()
                orig_img_deprocessed = orig_img.numpy()
                
                A = gen_img_deprocessed - orig_img_deprocessed
                                
                return 2, iters, gen_img_deprocessed, A
        
    gen_img_deprocessed = test_image_deprocess.numpy()
    orig_img_deprocessed = orig_img.numpy()

    A = gen_img_deprocessed - orig_img_deprocessed
    
    if label1 == orig_label and label1 == label2:
        return 1, -1,gen_img_deprocessed, A
        
    if label2 == orig_label and label1 != label2:
        return 3, -1,gen_img_deprocessed, A
        
    if label1 != orig_label and label2 != orig_label:
        return 4, -1,gen_img_deprocessed, A
        
        
    return 5, -1, gen_img_deprocessed, A

def calc_normal_success(method, CC='', CW='',WC='',WW='',OTHER=''):
    
    total=0
    badimg = 0
    
    cCC = 0
    cCW = 0
    cWC = 0
    cWW = 0
    cOTHER = 0
    
    for i, features in enumerate(mydataset):
        
        print(i)

        images = features['image']
        labels = features['label']

        for j,image in enumerate(images):
            
            label = labels[j].numpy()

            option, steps, gen, A = method(image,label)

            total += 1

            if option == 1:
                print("CC")
                np.save(locald +CC+"/"+"CC"+str(total)+".npy", gen)
                np.save(locald +CC+"/"+"CCfilter"+str(total)+".npy", A)
                cCC+=1
                
            
            if option == 2:
                print("CW")
                np.save(locald +CW+"/"+"CW"+str(total)+".npy", gen)
                np.save(locald +CW+"/"+"CWfilter"+str(total)+".npy", A)
                
                with open(locald +CW+"/"+"steps_data.csv", 'a') as f:
                    f.write(str(steps) + ", ")
                
                cCW+=1
                
            
            if option == 3:
                print("WC")
                np.save(locald +WC+"/"+"WC"+str(total)+".npy", gen)
                np.save(locald +WC+"/"+"WCfilter"+str(total)+".npy", A)
                cWC+=1
                
            
            if option == 4:
                print("WW")
                np.save(locald +WW+"/"+"WW"+str(total)+".npy", gen)
                np.save(locald +WW+"/"+"WWfilter"+str(total)+".npy", A)
                cWW+=1
                
            if option == 5:
                print("OTHER")
                np.save(locald +OTHER+"/"+"OTHER"+str(total)+".npy", gen)
                np.save(locald +OTHER+"/"+"OTHERfilter"+str(total)+".npy", A)
                cOTHER+=1
                
            
            if option == -2:
                badimg += 1
                total -= 1
                print("Bad Image",badimg)
                
                
            if option == -3:
                badimg += 1
                total -= 1
                print("Incorrect Image",badimg)
            

            print("CC:",cCC)
            print("CW:",cCW)
            print("WC:",cWC)
            print("WW:",cWW)
            print("OTHER:",cOTHER)

    print("Total:", total)
    print("BadImage:", badimg)
    print("CC:",cCC)
    print("CW:",cCW)
    print("WC:",cWC)
    print("WW:",cWW)
    print("OTHER:",cOTHER)


calc_normal_success(semi_bb,
                    CC='CC',
                    CW='CW',
                    WC='WC', WW='WW',OTHER='OTHER')