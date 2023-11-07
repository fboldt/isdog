from assessarchiteture import assessarchiteture
import tensorflow as tf
from tf.keras.applications.xception import Xception
from tf.keras.applications.vgg16 import VGG16

architetures = [
    Xception(input_shape = [image_size,image_size,3], weights='imagenet', include_top=False),
    VGG16(input_shape = [image_size,image_size,3], weights='imagenet', include_top=False),
]

for architeture in architetures:
    assessarchiteture(architeture)
