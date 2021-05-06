import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import enable_eager_execution
enable_eager_execution()
import tensorflow_datasets as tfds

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
    return features

BATCH_SIZE = 20000
tfds_dataset2, tfds_info  = tfds.load(name='imagenet2012_subset', split='validation[-40%:]', with_info=True,
                                     data_dir='/local/rcs/wei/image_net/')

val_ds = tfds_dataset2.map(preprocess_image).batch(BATCH_SIZE).prefetch(1)


for i, features in enumerate(val_ds):
    np.save('/local/rcs/wei/image_net/20000_imagnet_2012_lable.npy',features["label"].numpy())

