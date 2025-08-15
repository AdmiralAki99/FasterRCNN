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

from util import *

# Creating the backbone model for the Faster R-CNN architecture
# Using the VGG-16 architecture as the backbone

# VGG 16 backbone model

class VGG_16_NFCL(tf.keras.Model):
    def __init__(self):
        super(VGG_16_NFCL, self).__init__()

        # Layer 1
        self.conv_1a = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', name="block1_conv1")
        self.conv_1b = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', name="block1_conv2")
        self.max_pool_1a = MaxPool2D(pool_size=2, strides=2, padding='same', name="block1_pool")

        # Layer 2
        self.conv_2a = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', name="block2_conv1")
        self.conv_2b = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', name="block2_conv2")
        self.max_pool_2a = MaxPool2D(pool_size=2, strides=2, padding='same', name="block2_pool")

        # Layer 3
        self.conv_3a = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', name="block3_conv1")
        self.conv_3b = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', name="block3_conv2")
        self.conv_3c = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', name="block3_conv3")
        self.max_pool_3a = MaxPool2D(pool_size=2, strides=2, padding='same', name="block3_pool")

        # Layer 4
        self.conv_4a = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', name="block4_conv1")
        self.conv_4b = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', name="block4_conv2")
        self.conv_4c = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', name="block4_conv3")
        self.max_pool_4a = MaxPool2D(pool_size=2, strides=2, padding='same', name="block4_pool")

        # Layer 5
        self.conv_5a = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', name="block5_conv1")
        self.conv_5b = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', name="block5_conv2")
        self.conv_5c = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', name="block5_conv3")

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
        self.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',by_name=True)
        
        
        
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
        self.object_classfication_layer = Conv2D(num_of_anchors_per_pixel * 1,(1,1),name="Objectness Layer")

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
        object_scores = tf.reshape(object_scores,[object_scores.shape[0],object_scores.shape[1],object_scores.shape[2],self.num_of_anchors_per_pixel,1])
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
        object_scores = tf.keras.layers.Reshape((self.feature_map_height, self.feature_map_width, self.num_of_anchors_per_pixel, 1),name="Objectness Score Reshape")(object_scores)
        offsets = tf.keras.layers.Reshape((self.feature_map_height, self.feature_map_width, self.num_of_anchors_per_pixel, 4),name="Bounding Box Offsets Reshape")(offsets)

        return tf.keras.Model(inputs=[x],outputs=[object_scores,offsets])
    
    
# Region of Interest (RoI) Head Layer

# Function to pass the RoI's to become flattened to be passed through the connected layers
class RoIHead(Layer):
    def __init__(self,num_of_classes):
        super(RoIHead,self).__init__()

        # Defining the layers based on the paper
        self.num_of_classes = num_of_classes + 1 # Needs to include the background class as well (n +1)
        self.flatten = Flatten()
        self.fc_layer_1 = Dense(1024,activation='relu')
        self.dropout = Dropout(0.5) # Dropout layer to stop over fitting
        self.fc_layer_2 = Dense(1024,activation='relu')
        self.classifier_head = Dense((self.num_of_classes)) # Outputs the probabilities of the different classes
        self.regression_head = Dense((self.num_of_classes * 4)) # Coordinates for the bounding box (Need to make sure this is correct)

    def call(self, inputs):
        # Pass the input through the layer
        inputs = self.flatten(inputs)
        inputs = self.fc_layer_1(inputs)
        inputs = self.dropout(inputs)
        inputs = self.fc_layer_2(inputs)
        logits = self.classifier_head(inputs)
        deltas = self.regression_head(inputs)

        return logits, deltas # Returning the class output logits as well as the regression coordinates for the bounding box
    
# Consolidated Faster R-CNN Model
# class FasterRCNN(tf.keras.Model):
    
#     def __init__(self, num_of_anchors_per_pixel = 20, num_classes = 20):
        
#         super(FasterRCNN, self).__init__()
#         # Creating the backbone for the model
#         self.backbone = VGG_16_NFCL()
#         # Creating the RPN for the model
#         self.rpn = RegionProposalNetwork(num_of_anchors_per_pixel)
#         # Creating the RoI Head for the model
#         self.roi_head = RoIHead(num_classes)

#         self.num_classes = num_classes + 1

#         self.roi_delta_loss = Huber(delta=1.0)

#     def call(self,images,gt_boxes = None, gt_labels = None, training=False):
#         # Pass the image through the backbone
#         feature_map = self.backbone.call(images)

#         # Initialize the anchors for the model
#         anchor_boxes = initialize_all_anchor_boxes(images.shape, feature_map.shape)
        
#         # Pass the feature map to the RPN
#         objectness_scores,bbox_deltas = self.rpn.call(feature_map)
        
#         # Debugging information
#         tf.print("Feature Map Shape:", feature_map.shape)
#         tf.print("Anchor Boxes Shape:", anchor_boxes.shape)
#         tf.print("Objectness Scores Shape:", objectness_scores.shape)
#         tf.print("Bounding Box Deltas Shape:", bbox_deltas.shape)
        
#         # Refine region of interests
#         proposals = refine_region_of_interests_inference(anchor_boxes,bbox_deltas)

#         # Flattening the proposals 
#         B,H,W,A,_ = proposals.shape

#         proposals_flattened = tf.reshape(proposals,[B,-1,4])
        
#         # Scale the proposals to the original image size
#         stride = tf.cast(images.shape[1] / feature_map.shape[1], tf.float32)
        
#         proposals_flattened = proposals_flattened * stride

#         if training:

#             # Compute IoU and object labels
#             object_labels, iou_matrix = generate_objectness_labels(anchor_boxes, gt_boxes)

#             # Calculating the RPN Loss
#             rpn_loss = calculate_rpn_loss(anchor_boxes,bbox_deltas,gt_boxes,iou_matrix,object_labels,objectness_scores)

#             # Getting positive anchors for RoI
#             pos_anchors, pos_offsets, pos_indices = get_positive_anchor_boxes_and_corresponding_offsets(anchor_boxes, bbox_deltas, object_labels)

#             # RoI Pooling
#             roi_blocks, roi_boxes = roi_pooling(feature_map, pos_anchors, pos_indices, pos_offsets)

#             # Assign RoI to GT Boxes

#             roi_labels, best_gt_indices, max_iou_per_anchor = assign_roi_to_ground_truth_box(gt_boxes, roi_boxes, gt_labels)

#             # Match Ground truth box to the RoI Coordinate
#             matched_gt_boxes = match_gt_box_to_roi_coordinate(gt_boxes,best_gt_indices)

#             # Calculating the deltas for the RoI

#             roi_targets, filtered_roi_labels = calculate_bounding_box_deltas_between_roi_and_ground_truth_box(matched_gt_boxes, roi_boxes, roi_labels)
            
#             tf.print("Labels Shape Pre Sampling:", tf.shape(filtered_roi_labels))
#             tf.print("BBoxes Shape Pre Sampling:", tf.shape(roi_boxes))
#             tf.print("Deltas Shape Pre Sampling:", tf.shape(roi_targets))
    
#             tf.print("Labels Pre Sampling:", filtered_roi_labels)
#             tf.print("BBoxes Pre Sampling:", roi_boxes)
#             tf.print("Deltas Pre Sampling:", roi_targets)
            
#             # # Sampling RoI Blocks
            
#             # sampled_roi_boxes = []
#             # sampled_roi_labels = []
#             # sampled_roi_targets = []
#             # sampled_roi_blocks = []
            
#             # for batch in range(B):
#             #     # Get the sample RoIs per image
#             #     roi_boxes, roi_label,roi_target,roi_blocks = sample_rois_per_image(roi_boxes[batch], filtered_roi_labels[batch], roi_targets[batch], roi_blocks[batch])
#             #     sampled_roi_boxes.append(roi_boxes)
#             #     sampled_roi_labels.append(roi_label)
#             #     sampled_roi_targets.append(roi_target)
#             #     sampled_roi_blocks.append(roi_blocks)
                
#             # # Converting the lists to tensors
#             # roi_boxes = tf.stack(sampled_roi_boxes, axis=0)
#             # filtered_roi_labels = tf.stack(sampled_roi_labels, axis=0)
#             # roi_targets = tf.stack(sampled_roi_targets, axis=0)
#             # roi_blocks = tf.stack(sampled_roi_blocks, axis=0)
            
#             # tf.print("Labels Post Sampling:", filtered_roi_labels.shape)
#             # tf.print("RoI Blocks Post Sampling:", roi_boxes.shape)
#             # tf.print("Deltas Post Sampling:", roi_targets.shape)
#             # tf.print("RoI Blocks Shape:", roi_blocks.shape)
            
#             # RoI Head

#             class_scores, roi_bbox_deltas = self.roi_head(roi_blocks)
            
#             # tf.print("Filtered ROI labels:", filtered_roi_labels)
            
#             # assert tf.reduce_all((filtered_roi_labels >= 0) & (filtered_roi_labels < self.num_classes)), "Invalid class labels"
            
            
#             tf.debugging.check_numerics(class_scores, "Class scores contain NaN or Inf")

#             valid_mask = tf.where((filtered_roi_labels >= 0) & (filtered_roi_labels < self.num_classes))[:, 0]
#             valid_labels = tf.gather(filtered_roi_labels, valid_mask)
#             valid_logits = tf.gather(class_scores, valid_mask)

#             if tf.shape(valid_labels)[0] > 0:
#                 classification_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(
#                     valid_labels, valid_logits
#                 )
#             else:
#                 classification_loss = tf.constant(0.0)
            
#             regression_loss = calculate_roi_head_regression_loss(roi_targets,roi_bbox_deltas,filtered_roi_labels,number_of_classes=self.num_classes)

#             total_loss = rpn_loss  + classification_loss + regression_loss
            
#             return {
#                 "feature_map": feature_map,
#                 "anchors": anchor_boxes,
#                 "proposals" : proposals_flattened,
#                 "rpn_loss": rpn_loss,
#                 "rpn_objectness_score": objectness_scores,
#                 "rpn_deltas": bbox_deltas,
#                 "roi_cls_loss": classification_loss,
#                 "roi_reg_loss": regression_loss,
#                 "roi_deltas": roi_bbox_deltas,
#                 "total_loss": total_loss
#             }
            
#         # Inference Logic

#         # Filtering the top k boxes
#         top_proposals_flattened,top_indices_flattened = inference_filter_top_k(proposals_flattened,objectness_scores)
             
#         # Converting the proposals to center coordinates
#         top_proposals_xy_px  = convert_center_format_boxes_to_xy_coordinate(top_proposals_flattened)

#         # Scaling the top proposals to the feature map size
#         top_proposals_xy_fm = top_proposals_xy_px / stride

#         # RoI pooling inference
#         roi_proposals = roi_pooling_inference(feature_map,top_proposals_xy_fm,top_indices_flattened[:,0])
        
#         # Passing it through the RoI Head
#         classification_score, regression_head = self.roi_head(roi_proposals)
        
#         # Calculating the probabilities of the classes
#         classification_probs = tf.nn.softmax(classification_score, axis=-1)
        
#         # Foreground probabilities
#         foreground_probs = classification_probs[:,1:]

#         # Calculating the predicted labels
#         final_labels = tf.argmax(foreground_probs,axis=-1) + 1 # Adding 1 to account for the background class
        
#         # Converting proposals to xy coordinates
#         top_proposals_flattened = convert_center_format_boxes_to_xy_coordinate(top_proposals_flattened)
        
#         foreground_proposals_px,foreground_scores,foreground_labels,foreground_offsets_filtered = filter_foreground_predictions(top_proposals_xy_px,classification_probs,regression_head,final_labels,self.num_classes)

#         # Converting the xy-coordinates to center coordinates
#         foreground_proposals_center = convert_xy_boxes_to_center_format(foreground_proposals_px)
        
#         # Applying the offsets to the filtered proposals

#         adjusted_foreground_proposals = apply_bounding_box_deltas(foreground_proposals_center,foreground_offsets_filtered)

#         # Applying Non Max Suppression (NMS)
        
#         # indices = tf.stack([tf.range(tf.shape(classification_score)[0],dtype=tf.int64), final_labels], axis=1)
#         # predicted_class_scores = tf.gather_nd(classification_score, indices)
        
#         final_boxes = []
#         final_scores = []
#         final_labels = []

#         for class_id in range(1,self.num_classes):
#             # Creating a binary mask for each class
#             class_mask = foreground_labels == class_id

#             # Checking if there are no boxes for the class
#             if not tf.reduce_any(class_mask):
#                 continue

#             # Getting the for the class
#             boxes_per_class = tf.boolean_mask(adjusted_foreground_proposals,class_mask)
#             scores_per_class = tf.boolean_mask(foreground_scores,class_mask)
            
#             print("boxes shape:", boxes_per_class.shape)
#             print("scores shape:", scores_per_class.shape)
#             print("max score:", tf.reduce_max(scores_per_class))
#             print("min score:", tf.reduce_min(scores_per_class))
#             print("num scores > 0.02:", tf.reduce_sum(tf.cast(scores_per_class > 0.02, tf.int32)))


#             if boxes_per_class.shape[0] == 0 or scores_per_class.shape[0] == 0:
#                 continue
            
#             # Applying NMS for each class
#             class_indices = tf.image.non_max_suppression(boxes = boxes_per_class,scores=scores_per_class,max_output_size = 100,iou_threshold = 0.5,score_threshold = 0.02)

#             # Gathering boxes, scores, labels
#             nms_boxes = tf.gather(boxes_per_class,class_indices)
#             nms_scores = tf.gather(scores_per_class,class_indices)
#             nms_labels = tf.fill(tf.shape(nms_scores),class_id)

#             # Appending the boxes
#             final_boxes.append(nms_boxes)
#             final_scores.append(nms_scores)
#             final_labels.append(nms_labels)
       
#         if final_boxes:
#             final_boxes = tf.concat(final_boxes,axis=0)
#             final_scores = tf.concat(final_scores,axis=0)
#             final_labels = tf.concat(final_labels,axis=0)
#         else:
#             final_boxes = tf.zeros([0, 4])
#             final_scores = tf.zeros([0])
#             final_labels = tf.zeros([0], dtype=tf.int64)
        
#         return {
#         "boxes": final_boxes,
#         "scores": final_scores,
#         "labels": final_labels
#         }
          
# TODO: Implement the training loop for Tensorboard logging

# TODO: Implement the new Pipelines from me

class FasterRCNN(tf.keras.Model):
    
    def __init__(self, num_of_anchors_per_pixel = 20, num_classes = 20):
        
        super(FasterRCNN, self).__init__()
        # Creating the backbone for the model
        self.backbone = VGG_16_NFCL()
        # Creating the RPN for the model
        self.rpn = RegionProposalNetwork(num_of_anchors_per_pixel)
        # Creating the RoI Head for the model
        self.roi_head = RoIHead(num_classes)

        self.num_classes = num_classes + 1

        self.roi_delta_loss = Huber(delta=1.0)

    def call(self,images,gt_boxes = None, gt_labels = None, training=False, debug=False):
        # Pass the image through the backbone
        feature_map = self.backbone.call(images)
        
        # Initialize the anchor boxes for the model
        anchors = initialize_all_anchor_boxes(images.shape, feature_map.shape)
        
        # Isolating the training and inference pipelines
        if training:
            # Training Logic
            
            # Pass the feature map to the RPN
            objectness_scores, bounding_box_deltas = self.rpn.call(feature_map)
            
            # Debugging information
            if debug:
                tf.print("Feature Map Shape:", feature_map.shape)
                tf.print("Anchor Boxes Shape:", anchors.shape)
                tf.print("Objectness Scores Shape:", objectness_scores.shape)
                tf.print("Bounding Box Deltas Shape:", bounding_box_deltas.shape)
                
            # Generate objectness labels and IoU matrix
            object_labels, iou_matrix = generate_objectness_labels(anchors, gt_boxes)
            
            # Calculate the RPN Loss
            
            
            
        else:
            # Inference Logic
            
            pass