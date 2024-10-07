import tensorflow as tf
import numpy as np
from keras.api.preprocessing.image import ImageDataGenerator
from keras.api.layers import Input, Conv2D, MaxPool2D, BatchNormalization, Activation, Add, ZeroPadding2D,AveragePooling2D

"""
    VGG 16 Model with No Fully Connected Layers present. Used primarily as the backbone for the Mask RCNN  
"""
class VGG_16_NFCL(tf.keras.Model):
    def __init__(self):
        super(VGG_16_NFCL, self).__init__()

        # Layer 1
        self.conv_1a = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', name="conv_1A")
        self.conv_1b = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', name="conv_1B")
        self.max_pool_1a = MaxPool2D(pool_size=2, strides=2, padding='same', name="max_pool1A")

        # Layer 2
        self.conv_2a = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', name="conv_2A")
        self.conv_2b = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', name="conv_2B")
        self.max_pool_2a = MaxPool2D(pool_size=2, strides=2, padding='same', name="max_pool2A")

        # Layer 3
        self.conv_3a = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', name="conv_3A")
        self.conv_3b = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', name="conv_3B")
        self.conv_3c = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', name="feature_map_1")
        self.max_pool_3a = MaxPool2D(pool_size=2, strides=2, padding='same', name="max_pool3A")

        # Layer 4
        self.conv_4a = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', name="conv_4A")
        self.conv_4b = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', name="conv_4B")
        self.conv_4c = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', name="feature_map_2")
        self.max_pool_4a = MaxPool2D(pool_size=2, strides=2, padding='same', name="max_pool4A")

        # Layer 5
        self.conv_5a = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', name="conv_5A")
        self.conv_5b = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', name="conv_5B")
        self.conv_5c = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', name="feature_map_3")

    def call(self, input_tensor, training=False, mask=None):
        x = self.conv_1a(input_tensor)
        x = self.conv_1b(x)
        x = self.max_pool_1a(x)

        x = self.conv_2a(x)
        x = self.conv_2b(x)
        x = self.max_pool_2a(x)

        x = self.conv_3a(x)
        x = self.conv_3b(x)
        x = self.conv_3c(x)
        x = self.max_pool_3a(x)

        x = self.conv_4a(x)
        x = self.conv_4b(x)
        x = self.conv_4c(x)
        x = self.max_pool_4a(x)

        x = self.conv_5a(x)
        x = self.conv_5b(x)
        x = self.conv_5c(x)

        return x

    def build_graph(self,input_size):
        x = tf.keras.layers.Input(shape=(input_size[0],input_size[1],3))
        return tf.keras.Model(inputs=[x],outputs=self.call(x))

    def build(self,input_shape):
        super().build(input_shape)

class Faster_RCNN(tf.keras.Model):
    def __init__(self) -> None:
        pass
    
    
    
    