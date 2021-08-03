import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="5"
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
from SemibbModel import *


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

mode = 'r'
BATCH_SIZE = 50
c = 1
grad_iterations = 20
step = 1
epsilon = 8

# input image dimensions
img_rows, img_cols = 224 ,224

es = {'file_name': tf.TensorSpec(shape=(), dtype=tf.string, name=None),
 'image': tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32, name=None),
 'label': tf.TensorSpec(shape=(), dtype=tf.int64, name=None)}

mydataset = tf.data.experimental.load("/local/rcs/wei/End3kImagePerClass/",es).batch(BATCH_SIZE).prefetch(1)



if mode == 'm':
    model_ = tf.keras.applications.MobileNet(input_shape= (img_rows, img_cols,3))
    q_model = tfmot.quantization.keras.quantize_model(model_)
    model = tf.keras.applications.MobileNet(input_shape= (img_rows, img_cols,3))
    d_model = tf.keras.applications.MobileNet(input_tensor = q_model.input)
    model.load_weights("./fp_model_40_mobilenet.h5")
    q_model.load_weights("./q_model_40_mobilenet.h5")
    d_model.load_weights("./distilled_fp_model_40_mobilenet.h5")
    model.trainable = False
    q_model.trainable = False
    d_model.trainable = False
    preprocess = tf.keras.applications.mobilenet.preprocess_input
    decode = tf.keras.applications.mobilenet.decode_predictions
    net = 'mobile'

elif mode == 'r':
    model_ = ResNet50(input_shape= (img_rows, img_cols,3))
    q_model = tfmot.quantization.keras.quantize_model(model_)
    model = ResNet50(input_shape= (img_rows, img_cols,3))
    model.load_weights("./fp_model_40_resnet50.h5")
    q_model.load_weights("./q_model_40_resnet50.h5")
    s_model = buildmodel('fp')
    dq_model= tfmot.quantization.keras.quantize_model(buildmodel('fp'))
    s_model.load_weights("./segregated_fp_model_40_resnet.h5")
    dq_model.load_weights("./differentialble_q_model_40_resnet.h5")
    model.trainable = False
    q_model.trainable = False
    s_model.trainable = False
    dq_model.trainable = False
    preprocess = tf.keras.applications.resnet.preprocess_input
    decode = tf.keras.applications.resnet.decode_predictions
    net = 'res'

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

    model = tf.keras.applications.DenseNet121(input_shape= (img_rows, img_cols,3))
    d_model = tf.keras.applications.DenseNet121(input_tensor = q_model.input)
    model.load_weights("./fp_model_40_densenet121.h5")
    q_model.load_weights("./q_model_40_densenet121.h5")
    d_model.load_weights("./distilled_fp_model_40_densenet121.h5")
    model.trainable = False
    q_model.trainable = False
    d_model.trainable = False
    preprocess = tf.keras.applications.densenet.preprocess_input
    decode = tf.keras.applications.densenet.decode_predictions
    net = 'dense'



def second(image,label):
    orig_img = tf.identity(image)
    input_image = tf.identity(image)
    orig_logist = tf.identity(model.predict(preprocess(input_image)[None,...]) )
    orig_label =  np.argmax(orig_logist[0])

    
    quant_logist = tf.identity(dq_model.predict(preprocess(input_image)[None,...]))
    quant_label =  np.argmax(quant_logist[0])

    s_logist =  tf.identity(s_model.predict(preprocess(input_image)[None,...]))
    s_label =  np.argmax(s_logist[0])

    
    if orig_label != quant_label or orig_label != s_label:
        return -2,-2,-2,-2,-2
    
    if orig_label != label:
        return -3,-3,-3,-3,-3
    
    A = 0
    start_time = time.time()
    for iters in range(0,grad_iterations):
        with tf.GradientTape() as g:
            g.watch(input_image)
            loss1 = K.mean(s_model(preprocess(input_image)[None,...], training = False)[..., orig_label])
            loss2 = K.mean(dq_model(preprocess(input_image)[None,...], training = False)[..., orig_label])
            final_loss = K.mean(loss1 - c*loss2)


        grads = normalize(g.gradient(final_loss, input_image))
        adv_image = input_image + tf.sign(grads) * step
        A = tf.clip_by_value(adv_image - orig_img, -epsilon, epsilon)
        input_image = tf.clip_by_value(orig_img + A, 0, 255)
        test_image = preprocess(input_image)[None,...]
        pred1, pred2= model.predict(test_image), q_model.predict(test_image)
        label1, label2 = np.argmax(pred1[0]), np.argmax(pred2[0])
        pred3, pred4= s_model.predict(test_image), dq_model.predict(test_image)
        label3, label4 = np.argmax(pred3[0]), np.argmax(pred4[0])
        
        if not label3 == label4:
            if label3 == orig_label and decode(pred3, top=1)[0][0][2] > 0.6:

                total_time = time.time() - start_time
                
                gen_img_deprocessed = input_image
                orig_img_deprocessed = orig_img
                A = (gen_img_deprocessed - orig_img_deprocessed).numpy()

                if label1 != orig_label or label1 == label2:
                    print("wtf")
                    return -1, -1, -1, gen_img_deprocessed, A
                
                norm = np.max(np.abs(A))
                
                return total_time, norm, iters, gen_img_deprocessed, A

    gen_img_deprocessed = input_image
    orig_img_deprocessed = orig_img
    A = (gen_img_deprocessed - orig_img_deprocessed).numpy()

    return -1, -1, -1, gen_img_deprocessed, A

def topk(model_pred, qmodel_pred, k):
    preds = decode(model_pred, top=k)
    qpreds = decode(qmodel_pred, top=1)[0][0][1]
    
    for pred in preds[0]:
        if pred[1] == qpreds:
            return True
    
    return False

def secondk(image,k):
    orig_img = tf.identity(image)
    input_image = tf.identity(image)
    orig_logist = tf.identity(model.predict(preprocess(input_image)[None,...]) )
    orig_label =  np.argmax(orig_logist[0])

    
    quant_logist = tf.identity(dq_model.predict(preprocess(input_image)[None,...]))
    quant_label =  np.argmax(quant_logist[0])

    s_logist =  tf.identity(s_model.predict(preprocess(input_image)[None,...]))
    s_label =  np.argmax(s_logist[0])

    
    if orig_label != quant_label or orig_label != s_label:
        return -2,-2,-2,-2,-2
    
    A = 0
    start_time = time.time()
    for iters in range(0,grad_iterations):
        with tf.GradientTape() as g:
            g.watch(input_image)
            loss1 = K.mean(s_model(preprocess(input_image)[None,...], training = False)[..., orig_label])
            loss2 = K.mean(dq_model(preprocess(input_image)[None,...], training = False)[..., orig_label])
            final_loss = K.mean(loss1 - c*loss2)


        grads = normalize(g.gradient(final_loss, input_image))
        adv_image = input_image + tf.sign(grads) * step
        A = tf.clip_by_value(adv_image - orig_img, -epsilon, epsilon)
        input_image = tf.clip_by_value(orig_img + A, 0, 255)
        test_image = preprocess(input_image)[None,...]
        pred1, pred2= model.predict(test_image), q_model.predict(test_image)
        label1, label2 = np.argmax(pred1[0]), np.argmax(pred2[0])
        pred3, pred4= s_model.predict(test_image), dq_model.predict(test_image)
        label3, label4 = np.argmax(pred3[0]), np.argmax(pred4[0])
        
        if not topk(pred3, pred4, k):
            if label3 == orig_label and decode(pred3, top=1)[0][0][2] > 0.6:

                total_time = time.time() - start_time
                gen_img_deprocessed = input_image
                orig_img_deprocessed = orig_img
                A = (gen_img_deprocessed - orig_img_deprocessed).numpy()
                    

                if label1 == orig_label and not topk(pred1, pred2, k):
                    norm = np.max(np.abs(A))
                    return total_time, norm, iters, gen_img_deprocessed, A

                else:
                    return -1, -1, -1, gen_img_deprocessed, A
    
    gen_img_deprocessed = input_image
    orig_img_deprocessed = orig_img
    A = (gen_img_deprocessed - orig_img_deprocessed).numpy()

    return -1, -1, -1, gen_img_deprocessed, A

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
    failure = 0
    
    for i, features in enumerate(ds):

        images = features['image']
        labels = features['label']

        for j,image in enumerate(images):
            
            label = labels[j].numpy()

            time, advdist, steps, gen, A = method(image,label)

            total += 1

            if time == -1:
                print("Didnt find anything")
                np.save(locald + 'failure/' + folderName+"/"+dataName+str(failure)+"@"+str(total)+".npy", gen)
                np.save(locald + 'failure/' + filterName+"/"+dataName+str(failure)+"@"+str(total)+".npy", A)
                failure +=1
                continue
            
            if time == -2:
                badimg += 1
                total -= 1
                failure +=1
                print("Bad Image",badimg)
                continue
                
            if time == -3:
                badimg += 1
                total -= 1
                failure +=1
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
                np.save(locald + 'failure/' + folderName+"/"+dataName+"k"+str(failure)+".npy", gen)
                np.save(locald + 'failure/' + filterName+"/"+ dataName+"k"+str(failure)+".npy", A)
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

    print("Number seen:",total)
    print("No. worked:", count)
    print("No. topk:", top5)



calc_normal_success(second,secondk,mydataset,
                   folderName=net + 'net_imagenet_images_second', filterName=net +'net_imagenet_filters_second',dataName='second', dataFolder=net +'net_imagenet_data_second', locald ='/local/rcs/wei/V2/SemiBB_v2/' + net + 'net/' )