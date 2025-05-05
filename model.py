import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv
import numpy as np
import math

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, Activation, Add, ZeroPadding2D,AveragePooling2D
from tensorflow.keras.preprocessing.image import load_img,img_to_array

from tensorflow.keras.layers import Layer, Dense, Flatten, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy, Huber

# Creating the backbone model for the Faster R-CNN architecture
# Using the VGG-16 architecture as the backbone

# VGG 16 backbone model

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
        self.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
        
        
        
# Region Proposal Network (RPN) Model

class RegionProposalNetwork(tf.keras.Model):
    def __init__(self,num_of_anchors_per_pixel,**kwargs):
        """
        Region Proposal Network (RPN) as a subclass of Keras Model.
        
        Parameters:
        ----------
        num_of_anchors_per_pixel : int
            Number of anchor boxes per pixel (depends on the scales and aspect ratios used).
        """
        super(RegionProposalNetwork,self).__init__(**kwargs)

        # Initializing Info About Input Size & Number Of Anchors
        self.num_of_anchors_per_pixel = num_of_anchors_per_pixel
        self.feature_map_height = 50
        self.feature_map_width = 50

        # Creating the convolutional 3x3 layer of the architecture
        self.conv_layer = Conv2D(512,(3,3),padding='same',activation='relu',name="3x3 Conv Layer")

        # Creating Binary Classification Layer To Predict Objects/Background
        self.object_classfication_layer = Conv2D(num_of_anchors_per_pixel * 2,(1,1),activation='softmax',name="Objectness Layer")

        # Creating Bounding Box Regression Layer To Calculate Offsets For Anchor Boxes
        self.bounding_box_regression_layer = Conv2D(num_of_anchors_per_pixel * 4, (1,1),name="Bounding Box Regression Layer")

    def call(self,input_tensor):
        """
        Forward Pass For The Model Using Tensor Flow Model API

        Parameters:
        ---------
        input_tensor : Tensor
            Feature Maps Of Images

        Returns:
        -------
        object_scores: Tensor
            Predicted Scores Of Anchor Boxes Of Either Being Object/Background Made By Model

        offsets: Tensor
            Coordinates For Bounding Boxes Made By Model
        """

        # Need To Pass It Through The 3x3 Conv Layer
        conv_tensor = self.conv_layer(input_tensor)

        # Calculating The Objectness Score & Offset Regression Separately
        object_scores = self.object_classfication_layer(conv_tensor)
        offsets = self.bounding_box_regression_layer(conv_tensor)

        # reshaping the tensors to make into easier format
        object_scores = tf.reshape(object_scores,[object_scores.shape[0],object_scores.shape[1],object_scores.shape[2],self.num_of_anchors_per_pixel,2])
        offsets = tf.reshape(offsets,[offsets.shape[0],offsets.shape[1],offsets.shape[2],self.num_of_anchors_per_pixel,4])

        # We return the scores to the next part
        return object_scores, offsets

    def build_graph(self,input_size):
        """
        Build Graph Method For Examining Layer Architecture

        Parameters:
        ---------
        input_size : Tensor
            Feature Maps Of Images

        Returns:
        -------
           model: Tensorflow Model
               Tensorflow Model Of Region Proposal Network
        """
        x = tf.keras.layers.Input(shape=input_size)

        # Need To Pass It Through The 3x3 Conv Layer
        conv_tensor = self.conv_layer(x)

        # Calculating The Objectness Score & Offset Regression Separately
        object_scores = self.object_classfication_layer(conv_tensor)
        offsets = self.bounding_box_regression_layer(conv_tensor)

        # reshaping the tensors to make into easier format
        object_scores = tf.keras.layers.Reshape((self.feature_map_height, self.feature_map_width, self.num_of_anchors_per_pixel, 2),name="Objectness Score Reshape")(object_scores)
        offsets = tf.keras.layers.Reshape((self.feature_map_height, self.feature_map_width, self.num_of_anchors_per_pixel, 4),name="Bounding Box Offsets Reshape")(offsets)

        return tf.keras.Model(inputs=[x],outputs=[object_scores,offsets])
    
    
# Region of Interest (RoI) Head Layer

class RoIHead(Layer):
    def __init__(self,num_of_classes):
        super(RoIHead,self).__init__()

        # Defining the layers based on the paper
        self.num_of_classes = num_of_classes + 1 # Needs to include the background class as well (n +1)
        self.flatten = Flatten()
        self.fc_layer_1 = Dense(1024,activation='relu')
        self.dropout = Dropout(0.5) # Dropout layer to stop over fitting
        self.fc_layer_2 = Dense(1024,activation='relu')
        self.classifier_head = Dense((self.num_of_classes - 1),activation='softmax') # Outputs the probabilities of the different classes
        self.regression_head = Dense(4) # Coordinates for the bounding box (Need to make sure this is correct)

    def call(self, inputs):
        # Pass the input through the layer
        inputs = self.flatten(inputs)
        inputs = self.fc_layer_1(inputs)
        inputs = self.dropout(inputs)
        inputs = self.fc_layer_2(inputs)
        logits = self.classifier_head(inputs)
        deltas = self.regression_head(inputs)

        return logits, deltas # Returning the class output logits as well as the regression coordinates for the bounding box
    
    
# TODO: Implement the Consolidated Faster R-CNN Model

