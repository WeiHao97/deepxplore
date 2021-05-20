import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import PIL
from collections import Counter
import tensorflow as tf
import random
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.python.framework.ops import enable_eager_execution
enable_eager_execution()
from tensorflow.keras.layers import Input
import matplotlib.pyplot as plt
tf.config.list_physical_devices('GPU')
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow_datasets as tfds
from tensorflow.keras.losses import mean_squared_error, sparse_categorical_crossentropy,kl_divergence
import tensorflow_model_optimization as tfmot


def preprocess_image(features):
    """Preprocesses the given image.

      Args:
        image: `Tensor` representing an image of arbitrary size.

      Returns:
        A preprocessed image `Tensor` of range [0, 1].
  """
    image = features["image"]
    image = tf.image.resize(image,[224,224])
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.keras.applications.resnet.preprocess_input(image)
    
    features["image"] = image
    return features["image"], features["label"]

BATCH_SIZE = 256

tfds_dataset2, tfds_info  = tfds.load(name='imagenet2012_subset', split='validation[-60%:]', with_info=True,
                                     data_dir='/local/rcs/wei/image_net/')
val_ds = tfds_dataset2.map(preprocess_image).batch(BATCH_SIZE).prefetch(1)

img_rows, img_cols = 224 ,224
model = ResNet50(input_shape=(img_rows, img_cols,3))
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.load_weights("./original_model_40.h5")
with tf.device("/device:GPU:0"):
    model.evaluate(val_ds,verbose=1)
