import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import PIL
from PIL import Image
import tensorflow as tf
import random
import re
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.python.framework.ops import enable_eager_execution
#disable_eager_execution()
enable_eager_execution()
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow_datasets as tfds


def preprocess_input(x):
    x_temp = np.copy(x)
    x_temp = x_temp[..., ::-1]
    x_temp[..., 0] -= 91.4953
    x_temp[..., 1] -= 103.8827
    x_temp[..., 2] -= 131.0912
    return x_temp

tfds_dataset1 = tfds.load(name='lfw', split='train')
train_ds = tfds_dataset1.batch(50)
H, W, C = 224,224, 3
detector = MTCNN()
train_process = []
train_label = []
im_index = []
index = 0
for i, images in enumerate(train_ds):
    lb_index = 0
    if (i%100) == 0:
        np.save('/local/rcs/wei/FR/LFW_images.npy',np.array(train_process))
        np.save('/local/rcs/wei/FR/LFW_labels.npy',np.array(train_label))
        np.save('/local/rcs/wei/FR/LFW_index.npy',np.array(im_index))
        print(i)
    for image in images['image']:
        
        # detect faces in the image
        image_ = image.numpy()
        results = detector.detect_faces(image_)
        if len(results) != 0:
            # extract the bounding box from the first face
            x1, y1, width, height = results[0]['box']
            x2, y2 = x1 + width, y1 + height
            # extract the face
            if x1 >= 0 and y1 >= 0 and width >= 0 and height >= 0:
                #print(x1, y1, width, height)
                face = image_[y1:y2, x1:x2]
                # resize pixels to the model size
                face_array = tf.image.resize(face,[H,W]).numpy()
                img_ = preprocess_input(face_array[None,...])
                train_process.append(img_[0])
                train_label.append(images['label'][lb_index].numpy())
                im_index.append(index)
        lb_index += 1
        index += 1

np.save('/local/rcs/wei/FR/LFW_images.npy',np.array(train_process))
np.save('/local/rcs/wei/FR/LFW_labels.npy',np.array(train_label))
np.save('/local/rcs/wei/FR/LFW_index.npy',np.array(im_index))