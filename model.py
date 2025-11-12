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
    def __init__(self, name="backbone", **kwargs):
        super().__init__(name=name, **kwargs)

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
    def __init__(self,num_of_anchors_per_pixel,name="rpn",**kwargs):
        """
        Region Proposal Network (RPN) as a subclass of Keras Model.
        
        Parameters:
        ----------
        num_of_anchors_per_pixel : int
            Number of anchor boxes per pixel (depends on the scales and aspect ratios used).
        """
        super().__init__(name=name, **kwargs)

        # Initializing Info About Input Size & Number Of Anchors
        self.num_of_anchors_per_pixel = num_of_anchors_per_pixel
        self.feature_map_height = 50
        self.feature_map_width = 50

        # Creating the convolutional 3x3 layer of the architecture
        self.conv_layer = Conv2D(512,(3,3),padding='same',activation='relu',name="rpn_conv")

        # Creating Binary Classification Layer To Predict Objects/Background
        self.object_classfication_layer = Conv2D(num_of_anchors_per_pixel * 2,(1,1),bias_initializer = bias_init_for_prior(0.01),name="rpn_cls_logits")

        # Creating Bounding Box Regression Layer To Calculate Offsets For Anchor Boxes
        self.bounding_box_regression_layer = Conv2D(num_of_anchors_per_pixel * 4, (1,1),name="rpn_bbox_regression" )

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

        object_scores_softmax = tf.nn.softmax(object_scores,axis=-1)

        # We return the scores to the next part
        return object_scores_softmax, offsets

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

# Function to pass the RoI's to become flattened to be passed through the connected layers
class RoIHead(Layer):
    def __init__(self,num_of_classes, name="roi_head", **kwargs):
        super().__init__(name=name, **kwargs)

        # Defining the layers based on the paper
        self.num_of_classes = num_of_classes + 1 # Needs to include the background class as well (n +1)
        self.flatten = Flatten(name="roi_flatten")
        self.fc_layer_1 = Dense(1024,activation='relu',name="roi_fc1")
        self.dropout = Dropout(0.5) # Dropout layer to stop over fitting
        self.fc_layer_2 = Dense(1024,activation='relu',name="roi_fc2")
        self.classifier_head = Dense((self.num_of_classes),bias_initializer = bias_init_for_prior(0.01),name="roi_cls") # Outputs the logits of the different classes that needs to be passed through softmax
        self.regression_head = Dense((self.num_of_classes * 4), name="roi_bbox") # Coordinates for the bounding box for each class

    def call(self, inputs,training = False):

        # Reshape the RoIs to get the predictions for each RoI present
        shape = tf.shape(inputs)
        B, N, H, W, C = shape[0], shape[1], shape[2], shape[3], shape[4] 

        inputs = tf.reshape(inputs,[B*N,H,W,C])
        
        # Pass the input through the layer
        inputs = self.flatten(inputs)
        inputs = self.fc_layer_1(inputs)
        if training:
            inputs = self.dropout(inputs,training=True)
        inputs = self.fc_layer_2(inputs)
        logits = self.classifier_head(inputs)
        deltas = self.regression_head(inputs)

        # Reshaping the logits and deltas
        logits = tf.reshape(logits,[B,N,self.num_of_classes])
        deltas = tf.reshape(deltas,[B,N,self.num_of_classes * 4])

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

# TODO: Implement the metric functions into the model to track and add logging images into the tensorboard too.

class FasterRCNN(tf.keras.Model):
    
    def __init__(self, num_of_anchors_per_pixel = NUM_OF_ANCHORS_PER_PIXEL, num_classes = 20, name="faster_rcnn", **kwargs):
        
        super().__init__(name=name, **kwargs)
        # Creating the backbone for the model
        self.backbone = VGG_16_NFCL(name = "backbone")
        # Creating the RPN for the model
        self.rpn = RegionProposalNetwork(num_of_anchors_per_pixel,name="rpn")
        # Creating the RoI Head for the model
        self.roi_head = RoIHead(num_classes, name="roi_head")

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
            
            # Calculate the Objectness Loss for logging on Tensorboard
            objectness_loss = calculate_objectness_loss(objectness_scores[...,1],object_labels)
            
            # Calculate the Bounding Box Regression Loss for logging on Tensorboard
            bbox_regression_loss = calculate_bounding_box_regression_loss(anchors,bounding_box_deltas,gt_boxes,iou_matrix,object_labels)
            
            # Calculating the RPN Loss
            rpn_loss = calculate_rpn_loss(anchors,bounding_box_deltas,gt_boxes,iou_matrix,object_labels,objectness_scores)
            
            # Filtering Proposals for RoI Pooling Stage
            
            # Gather positive anchors and corresponding offsets
            # pos_anchors, pos_offsets, pos_indices = get_positive_anchor_boxes_and_corresponding_offsets(anchors, bounding_box_deltas, object_labels)
            
            # if debug:
            #     tf.print("Positive Anchors Shape:", pos_anchors.shape)
            #     tf.print("Positive Offsets Shape:", pos_offsets.shape)
            #     tf.print("Positive Indices Shape:", pos_indices.shape)
            
            # Create proposals from the RPN outputs
            proposals = create_proposals_from_rpn(anchors,bounding_box_deltas)
            
            # Filter proposals based on objectness scores
            proposals, scores = filter_proposals_by_size(proposals,objectness_scores[...,1])
            
            # Pre NMS Top-K Filtering
            pre_nms_top_k_proposals, pre_nms_top_k_scores = pre_nms_top_k(proposals,scores)
            
            if debug:
                tf.print("Pre NMS Top K Proposals Shape:", pre_nms_top_k_proposals.shape)
                tf.print("Pre NMS Top K Scores Shape:", pre_nms_top_k_scores.shape)
            
            # Applying NMS to the proposals
            nms_proposals, nms_scores = class_agnostic_nms_thresholding(pre_nms_top_k_proposals,pre_nms_top_k_scores)
            
            if debug:
                tf.print("NMS Proposals Shape:", nms_proposals.shape)
                tf.print("NMS Scores Shape:", nms_scores.shape)
                
            # Applying Post NMS Top-K Filtering
            post_nms_top_k_proposals, post_nms_top_k_scores = post_nms_top_k(nms_proposals,nms_scores)
            
            if debug:
                tf.print("Post NMS Top K Proposals Shape:", post_nms_top_k_proposals.shape)
                tf.print("Post NMS Top K Scores Shape:", post_nms_top_k_scores.shape)
                
            # Calculating the RPN health using Recall metric
            rpn_recall_per_img, rpn_recall_global = util_rpn_recall_post_nms(post_nms_top_k_proposals,gt_boxes,gt_labels)
            
            if debug:
                tf.print("RPN Health (Recall) Metric per Image:", rpn_recall_per_img)
                tf.print("RPN Health (Recall) Metric Global", rpn_recall_global)
                
            # # Roi Pooling Stage
            # roi_blocks, roi_coordinates = roi_pooling(feature_map, pos_anchors, pos_indices, pos_offsets)
            
            # if debug:
            #     tf.print("RoI Blocks Shape: ",roi_blocks.shape)
            #     tf.print("RoI Coordinates Shape",roi_coordinates.shape)
            
            # Sampling Proposals for RoI Align Stage
            sampled_proposals, sampled_scores = sample_proposals_per_image(post_nms_top_k_proposals,post_nms_top_k_scores,ground_truth_boxes= gt_boxes)
            
            if debug:
                tf.print("Sampled Proposals Shape:", sampled_proposals.shape)
                tf.print("Sampled Scores Shape:", sampled_scores.shape)
            
            # RoI Align Stage
            roi_blocks, roi_coordinates = roi_align(feature_map,sampled_proposals)
            
            if debug:
                tf.print("RoI Blocks Shape: ",roi_blocks.shape)
                tf.print("RoI Coordinates Shape",roi_coordinates.shape)
            
            # Assign RoI to GT Boxes
            roi_labels, best_gt_indices, max_iou_per_anchor = assign_roi_to_ground_truth_box(gt_boxes, roi_coordinates, gt_labels)
            
            # Calculating the health of the RoI Sampler
            roi_sampler_health = util_roi_sampler_health(roi_labels)
            
            if debug:
                tf.print("RoI Total:", roi_sampler_health['roi_total'])
                tf.print("RoI Positive Fraction Per Image:", roi_sampler_health['roi_pos_frac_per_image'])
                tf.print("RoI Positive Fraction Global:", roi_sampler_health['roi_pos_frac'])
                tf.print("RoI Positive Fraction Target Deviation:", roi_sampler_health['target_range_deviation'])
                tf.print("RoI Positive Fraction Deviation Per Image:", roi_sampler_health['pos_frac_deviation'])
            
            # Match Ground truth box to the RoI Coordinate
            matched_gt_boxes = match_gt_box_to_roi_coordinate(gt_boxes,best_gt_indices)
            
            # Calculate the bounding box deltas between RoI and ground truth boxes
            roi_targets, filtered_roi_labels = calculate_bounding_box_deltas_between_roi_and_ground_truth_box(matched_gt_boxes, roi_coordinates, roi_labels)
            
            # Debugging information
            if debug:
                tf.print("Labels Shape :", tf.shape(filtered_roi_labels))
                tf.print("BBoxes Shape:", tf.shape(roi_coordinates))
                tf.print("Deltas Shape:", tf.shape(roi_targets))
                
            # RoI Head Stage
            roi_classification_scores, roi_bbox_deltas = self.roi_head.call(roi_blocks,training=training)
            
            # Debugging information
            if debug:
                tf.debugging.check_numerics(roi_classification_scores, "Class scores contain NaN or Inf")
                tf.debugging.check_numerics(roi_bbox_deltas, "BBox deltas contain NaN or Inf")
                tf.print("ROI Classification Scores Shape:", roi_classification_scores.shape)
                tf.print("ROI BBox Deltas Shape:", roi_bbox_deltas.shape)
                
            tf.debugging.assert_equal(
                tf.shape(filtered_roi_labels)[1],   # [B, R_labels]
                tf.shape(roi_classification_scores)[1],  # [B, R_scores, C]
                message="RoI count mismatch between labels and scores"
            )
            
            # Calculating the RoI Regression Head Gain
            roi_head_regressor_gain = util_roi_head_regressor_gain(roi_coordinates,roi_labels,matched_gt_boxes,roi_bbox_deltas,gt_boxes,self.num_classes)
                
            # Calculate the RoI classification loss
            roi_classification_loss = calculate_roi_head_classification_loss(filtered_roi_labels,roi_classification_scores)
            
            # Calculate the regression loss
            roi_regression_loss = calculate_roi_head_regression_loss(roi_targets,roi_bbox_deltas,filtered_roi_labels,number_of_classes=self.num_classes)
            
            # Returning the losses and other information for logging
            return {
                "feature_map": feature_map,
                "anchors": anchors,
                "objectness_loss": objectness_loss,
                "bbox_regression_loss": bbox_regression_loss,
                "rpn_loss": rpn_loss,
                "roi_cls_loss": roi_classification_loss,
                "roi_reg_loss": roi_regression_loss,
                "total_loss": rpn_loss + roi_classification_loss + roi_regression_loss,
                "roi_head_regressor_gain": roi_head_regressor_gain.numpy(),
                "rpn_recall_global": rpn_recall_global.numpy(),
                "roi_sampler_pos_frac_global": roi_sampler_health['roi_pos_frac']        
            }                         
        else:
            # Inference Logic
            
            # Pass the feature map to the RPN
            objectness_scores, bounding_box_deltas = self.rpn.call(feature_map)
            
            # Debugging information
            # if debug:
            #     tf.print("Feature Map Shape:", feature_map.shape)
            #     tf.print("Anchor Boxes Shape:", anchors.shape)
            #     tf.print("Objectness Scores Shape:", objectness_scores.shape)
            #     tf.print("Bounding Box Deltas Shape:", bounding_box_deltas.shape)
                
            # Create Proposals from the RPN
            inference_proposals = inference_create_proposals(anchors=anchors,deltas=bounding_box_deltas)
            
            # if debug:
            #     tf.print("Inference Proposals Shape:", inference_proposals.shape)
            
            # Filter Proposals
            nms_proposals, nms_scores = inference_filter_proposals(inference_proposals,objectness_score=objectness_scores[...,1])
            
            # if debug:
            #     tf.print("NMS Proposals Shape:", nms_proposals.shape)
            #     tf.print("NMS Scores Shape:", nms_scores.shape)
            
            # RoI Align
            inference_rois = inference_roi_align(feature_map=feature_map,proposals=nms_proposals,scores=nms_scores)
            
            # if debug:
            #     tf.print("Inference RoI Shape:", inference_rois.shape)
                
            # RoI Head
            cls_scores, reg_head = self.roi_head.call(inference_rois,training=training)
            
            # if debug:
            #     tf.print("RoI Head Classification Scores Shape:", cls_scores.shape)
            #     tf.print("RoI Head Regression Head Shape:", reg_head.shape)
            
            # Create Valid Mask
            dummy_box =  tf.constant([0,0,1e-3,1e-3],dtype=nms_proposals[0].dtype)
            valid_mask = tf.logical_not(tf.reduce_all(tf.equal(nms_proposals,dummy_box[tf.newaxis,tf.newaxis,:]),axis=-1))
            
            # Calculate Classification Scores for RoIs
            pred_scores = inference_classification_decision(cls_scores,valid_mask=valid_mask)
            
            # if debug:
            #     tf.print("RoI Classification Scores:", pred_scores.shape)
            
            # Apply Bounding Box Offsets to RoI
            roi_proposals = inference_apply_deltas(nms_proposals,reg_head,self.num_classes)
                        
            # Combined NMS
            nms_boxes, nms_scores, nms_classes,valid_detections = inference_combined_non_max_suppression(roi_proposals,pred_scores,scores_thresh = 0.05,iou_thresh = 0.5,max_total_size = 100,max_output_size_per_class = 100 )
            
            if debug:
                return{
                    "Boxes": nms_boxes,
                    "Scores": nms_scores,
                    "Classes": nms_classes,
                    "Valid Detectons": valid_detections,
                    "Proposals": nms_proposals,
                }
            
            # Return the final boxes, scores, and labels
            return{
                "Boxes": nms_boxes,
                "Scores": nms_scores,
                "Classes": nms_classes,
                "Valid Detectons": valid_detections
            }