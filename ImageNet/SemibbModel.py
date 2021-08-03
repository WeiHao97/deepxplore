import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from keras_applications import get_submodules_from_kwargs
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from keras_applications.imagenet_utils import _obtain_input_shape

preprocess_input = imagenet_utils.preprocess_input

backend = None
layers = None
models = None
keras_utils = None

interpreter = tf.lite.Interpreter("./tflite_int8_model_40.tflite")
interpreter.allocate_tensors()
tensor_details = interpreter.get_tensor_details()
quanti_dict = {}
for dict_ in tensor_details:
    i = dict_['index']
    tensor_name = dict_['name']
    scales = dict_['quantization_parameters']['scales']
    zero_points = dict_['quantization_parameters']['zero_points']
    if tensor_name[-6:] == 'Conv2D':
        name = tensor_name[tensor_name[0:len(tensor_name)-7].rfind('/') + 1:len(tensor_name)-7]
        quanti_dict[name] = {}
        quanti_dict[name]['scales'] = scales[0]
        quanti_dict[name]['zero_points'] = zero_points[0]
   
    elif tensor_name[-7:] == 'Softmax':
        name = 'Softmax'
        quanti_dict[name] = {}
        quanti_dict[name]['scales'] = scales[0]
        quanti_dict[name]['zero_points'] = zero_points[0]
      
    elif tensor_name[-3:] == 'add':
        name = tensor_name[tensor_name[0:len(tensor_name)-4].rfind('/') + 1:len(tensor_name)-4]
        quanti_dict[name] = {}
        quanti_dict[name]['scales'] = scales[0]
        quanti_dict[name]['zero_points'] = zero_points[0]
     
    elif tensor_name[-25:] == 'quant_predictions/BiasAdd':
        name = 'FC'
        quanti_dict[name] = {}
        quanti_dict[name]['scales'] = scales[0]
        quanti_dict[name]['zero_points'] = zero_points[0]
     
    elif tensor_name[-4:] == 'Mean':
        name = 'Mean'
        quanti_dict[name] = {}
        quanti_dict[name]['scales'] = scales[0]
        quanti_dict[name]['zero_points'] = zero_points[0]
    
    elif tensor_name == 'resnet50/quant_conv1_pad/Pad':
        name = 'Input'
        quanti_dict[name] = {}
        quanti_dict[name]['scales'] = scales[0]
        quanti_dict[name]['zero_points'] = zero_points[0]
 


def Quant_layer(input_tensor,
                scale,
                zero_point):
    scales_r = np.full(backend.int_shape(input_tensor)[1:4], 1/scale,dtype= np.float32)[None,...]
    scales = np.full(backend.int_shape(input_tensor)[1:4], scale,dtype= np.float32)[None,...]
    zero_points = np.full(backend.int_shape(input_tensor)[1:4], zero_point,dtype= np.float32)[None,...]
    
    x = layers.Multiply()([input_tensor, scales_r])
    x = layers.add([x, zero_points])
    x = backend.cast(x,'int32')
    x = backend.clip(x, -128,127)
    x = backend.cast(x,'float32')
    x = layers.subtract([x, zero_points])
    x = layers.Multiply()([x, scales])
    return x

def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name = 'quant_conv' + str(stage) + '_block' + str(block) + '_1_conv'
    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name)(input_tensor)
    #x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)
    # s = quanti_dict[conv_name]['scales']
    # z = quanti_dict[conv_name]['scales']
    #x = Quant_layer(x,s,z)

    conv_name = 'quant_conv' + str(stage) + '_block' + str(block) + '_2_conv'
    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name)(x)
    #x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)
    # s = quanti_dict[conv_name]['scales']
    # z = quanti_dict[conv_name]['scales']
    #x = Quant_layer(x,s,z)

    conv_name = 'quant_conv' + str(stage) + '_block' + str(block) + '_3_conv'
    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name)(x)
    # s = quanti_dict[conv_name]['scales']
    # z = quanti_dict[conv_name]['scales']
    #x = Quant_layer(x,s,z)
    #x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    conv_name = 'quant_conv' + str(stage) + '_block1_0_conv'
    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name)(input_tensor)
    # s = quanti_dict[conv_name]['scales']
    # z = quanti_dict[conv_name]['scales']
    #shortcut = Quant_layer(shortcut,s,z)
    #shortcut = layers.BatchNormalization(
        #axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    conv_name = 'quant_conv' + str(stage) + '_block' + str(block) + '_add'
    # s = quanti_dict[conv_name]['scales']
    # z = quanti_dict[conv_name]['scales']
    #x = Quant_layer(x,s,z)
    return x

def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name = 'quant_conv' + str(stage) + '_block' + str(block) + '_1_conv'
    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name)(input_tensor)
    #x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)
    # s = quanti_dict[conv_name]['scales']
    # z = quanti_dict[conv_name]['scales']
    #x = Quant_layer(x,s,z)

    conv_name = 'quant_conv' + str(stage) + '_block' + str(block) + '_2_conv'
    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name)(x)
    #x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)
    # s = quanti_dict[conv_name]['scales']
    # z = quanti_dict[conv_name]['scales']
    #x = Quant_layer(x,s,z)

    conv_name = 'quant_conv' + str(stage) + '_block' + str(block) + '_3_conv'
    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name)(x)
    # s = quanti_dict[conv_name]['scales']
    # z = quanti_dict[conv_name]['scales']
    #x = Quant_layer(x,s,z)
    #x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    conv_name = 'quant_conv' + str(stage) + '_block' + str(block) + '_add'
    # s = quanti_dict[conv_name]['scales']
    # z = quanti_dict[conv_name]['scales']
    #x = Quant_layer(x,s,z)
    return x

def MyResNetQ(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):

    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = tf.keras.backend,tf.keras.layers,tf.keras.models,tf.keras.utils

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    
    x = Quant_layer(img_input,1.0774157,-13)
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal', 
                      name='quant_conv1_conv')(x)
    #x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    #x = Quant_layer(x,0.07133354,-90)
    
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='1', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='2')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='3')
    
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='1')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='2')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='3')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='4')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='1')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='2')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='3')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='4')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='5')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='6')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='1')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='2')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='3')
    
    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation = 'softmax',name='quant_predictions')(x)

    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='myresnet')

    return model


def conv_block_fp(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    #bn_name_base = 'bn' + str(stage) + block + '_branch'
    conv_name = 'quant_conv' + str(stage) + '_block' + str(block) + '_1_conv'
    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name)(input_tensor)
    #x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)
    # s = quanti_dict[conv_name]['scales']
    # z = quanti_dict[conv_name]['scales']
    #x = Quant_layer(x,s,z)

    conv_name = 'quant_conv' + str(stage) + '_block' + str(block) + '_2_conv'
    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name)(x)
    #x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)
    # s = quanti_dict[conv_name]['scales']
    # z = quanti_dict[conv_name]['scales']
    #x = Quant_layer(x,s,z)

    conv_name = 'quant_conv' + str(stage) + '_block' + str(block) + '_3_conv'
    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name)(x)
    # s = quanti_dict[conv_name]['scales']
    # z = quanti_dict[conv_name]['scales']
    #x = Quant_layer(x,s,z)
    #x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    conv_name = 'quant_conv' + str(stage) + '_block1_0_conv'
    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name)(input_tensor)
    # s = quanti_dict[conv_name]['scales']
    # z = quanti_dict[conv_name]['scales']
    #shortcut = Quant_layer(shortcut,s,z)
    #shortcut = layers.BatchNormalization(
        #axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    #conv_name = 'quant_conv' + str(stage) + '_block' + str(block) + '_add'
    # s = quanti_dict[conv_name]['scales']
    # z = quanti_dict[conv_name]['scales']
    #x = Quant_layer(x,s,z)
    return x

def identity_block_fp(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    #bn_name_base = 'bn' + str(stage) + block + '_branch'
    conv_name = 'quant_conv' + str(stage) + '_block' + str(block) + '_1_conv'
    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name)(input_tensor)
    #x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)
    # s = quanti_dict[conv_name]['scales']
    # z = quanti_dict[conv_name]['scales']
    #x = Quant_layer(x,s,z)

    conv_name = 'quant_conv' + str(stage) + '_block' + str(block) + '_2_conv'
    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name)(x)
    #x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)
    # s = quanti_dict[conv_name]['scales']
    # z = quanti_dict[conv_name]['scales']
    #x = Quant_layer(x,s,z)

    conv_name = 'quant_conv' + str(stage) + '_block' + str(block) + '_3_conv'
    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name)(x)
    # s = quanti_dict[conv_name]['scales']
    # z = quanti_dict[conv_name]['scales']
    #x = Quant_layer(x,s,z)
    #x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    conv_name = 'quant_conv' + str(stage) + '_block' + str(block) + '_add'
    # s = quanti_dict[conv_name]['scales']
    # z = quanti_dict[conv_name]['scales']
    #x = Quant_layer(x,s,z)
    return x

def MyResNetFP(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):

    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = tf.keras.backend,tf.keras.layers,tf.keras.models,tf.keras.utils

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal', 
                      name='quant_conv1_conv')(x)
    #x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block_fp(x, 3, [64, 64, 256], stage=2, block='1', strides=(1, 1))
    x = identity_block_fp(x, 3, [64, 64, 256], stage=2, block='2')
    x = identity_block_fp(x, 3, [64, 64, 256], stage=2, block='3')
    
    x = conv_block_fp(x, 3, [128, 128, 512], stage=3, block='1')
    x = identity_block_fp(x, 3, [128, 128, 512], stage=3, block='2')
    x = identity_block_fp(x, 3, [128, 128, 512], stage=3, block='3')
    x = identity_block_fp(x, 3, [128, 128, 512], stage=3, block='4')

    x = conv_block_fp(x, 3, [256, 256, 1024], stage=4, block='1')
    x = identity_block_fp(x, 3, [256, 256, 1024], stage=4, block='2')
    x = identity_block_fp(x, 3, [256, 256, 1024], stage=4, block='3')
    x = identity_block_fp(x, 3, [256, 256, 1024], stage=4, block='4')
    x = identity_block_fp(x, 3, [256, 256, 1024], stage=4, block='5')
    x = identity_block_fp(x, 3, [256, 256, 1024], stage=4, block='6')

    x = conv_block_fp(x, 3, [512, 512, 2048], stage=5, block='1')
    x = identity_block_fp(x, 3, [512, 512, 2048], stage=5, block='2')
    x = identity_block_fp(x, 3, [512, 512, 2048], stage=5, block='3')
    
    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation = 'softmax',name='quant_predictions')(x)

    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='myresnet')

    return model

def buildmodel(mode = 'q', input_t = None):
    
    if input_t == None:
        if mode == 'q':
            model = MyResNetQ(input_shape=(224, 224,3), weights=None)
        else:
            model = MyResNetFP(input_shape=(224, 224,3), weights=None)
    else:
        if mode == 'q':
            model = MyResNetQ(input_tensor=input_t, weights=None)
        else:
            model = MyResNetFP(input_tensor=input_t, weights=None)
    
    layernames = []
    for layer in model.layers:
        if 'quant_conv' in layer.name or 'quant_predictions' in layer.name:
            layernames.append(layer.name)

    weight_dict = {}
    for dict_ in tensor_details:
        i = dict_['index']
        tensor_name = dict_['name']
        scales = dict_['quantization_parameters']['scales']
        zero_points = dict_['quantization_parameters']['zero_points']
        intweight = interpreter.tensor(i)().copy()
        for layername in layernames:
            if layername in tensor_name:
                if 'quant_predictions/BiasAdd/ReadVariableOp/resource' in tensor_name:
                    weight_dict['quant_predictions/bias:0'] = np.multiply(np.subtract(intweight, zero_points),scales)
                    #weight_dict['quant_predictions/bias:0'] = intweight
                elif 'quant_predictions/LastValueQuant' in tensor_name:
                    intweight_ = np.moveaxis(intweight, 0, len(intweight.shape)-1)
                    weight_dict['quant_predictions/kernel:0'] = (intweight_ - zero_points[0]) *  scales[0]
                    #weight_dict['quant_predictions/kernel:0'] = intweight_
                
                else:
                    if 'quant_predictions' not in tensor_name and tensor_name.count(layername) ==2:
                        if 'FakeQuantWithMinMaxVarsPerChannel' not in tensor_name:
                            weight_dict[layername + '/bias:0']  = (intweight - zero_points) *  scales
                            #weight_dict[layername + '/bias:0']  = intweight
                        else:
                            intweight_ = intweight - zero_points.reshape((-1,) + (1,) * (intweight.ndim-1))
                            intweight_ = intweight_ * scales.reshape((-1,) + (1,) * (intweight.ndim-1))
                            #intweight_ = intweight
                            w = intweight
                            weight_dict[layername + '/kernel:0']  = np.moveaxis(intweight_, 0, len(intweight_.shape)-1)
    for layer in layernames:
        kernel_name = layer + '/kernel:0'
        bias_name = layer + '/bias:0'
        model.get_layer(layer).set_weights( [weight_dict[kernel_name],weight_dict[bias_name]])
        

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    return model 

class Distiller(Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results
