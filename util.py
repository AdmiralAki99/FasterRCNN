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
from tensorflow.keras.losses import SparseCategoricalCrossentropy, Huber, BinaryCrossentropy

from helpers import dump_tensor_to_txt

import matplotlib.colors as mcolors
import random

## CONSTANTS

"""
BASE_ANCHOR_BOX_SIZE = [128, 128]
ANCHOR_BOX_RATIOS = [0.333, 0.5, 1, 1.5]
ANCHOR_BOX_SCALES = [1, 2, 3, 4, 6, 7]

"""

BASE_ANCHOR_BOX_SIZE = [128, 128]
ANCHOR_BOX_RATIOS = [0.333, 0.5, 1, 1.5, 2.0]
ANCHOR_BOX_SCALES = [0.75, 1, 2, 3, 4, 6, 7] # TODO: These ratios are causing bigger boxes to be prioiritized early. [1, 2, 3, 4, 6, 7]

# Hyperparameters
RPN_POSITIVE_ANCHOR_IOU_THRESHOLD = 0.50
RPN_NEGATIVE_ANCHOR_IOU_THRESHOLD = 0.20
RPN_TOTAL_ANCHOR_SAMPLES = 256
RPN_POSITIVE_ANCHOR_FRACTION = 0.5
RPN_PRE_NMS_TOP_K = 12000
RPN_POST_NMS_TOP_K = 2000
PROPOSAL_MIN_SIZE = 1
NUM_OF_ANCHORS_PER_PIXEL = len(ANCHOR_BOX_RATIOS) * len(ANCHOR_BOX_SCALES)

# RoI Hyperparameters
ROI_SAMPLE_SIZE = 512
ROI_POSITIVE_FRACTION = 0.25
# IOU Threshold for positive proposals
ROI_POSITIVE_IOU_THRESHOLD = 0.4
# IOU Threshold for negative proposals
ROI_NEGATIVE_IOU_THRESHOLD = 0.0

# Function to get number of anchor points on a feature map
def get_number_of_anchor_points(feature_map) -> tuple[int, int, int]:
    """
    Calculates the number of anchor points (centers) for the the feature_map

    Parameters
    ----------
    feature_map: Feature Map created by CNN backbone

    Returns
    ----------
    Tuple[int, int, int]
        Tuple of the Number of anchor points, the X-axis size of the feature map, the Y-axis size of the feature map

    """

    if len(feature_map.shape) != 4:
        raise ValueError("Input must be a 4D tensor with shape (batch_size, height, width, channels)")
    
    # Get the Shape of the Image (WxH)
    _, axis_1, axis_2, _ = feature_map.shape
    # Total Number of Anchor Points Possible is WxH
    anchors = axis_1 * axis_2
    # Return Tuple of Number of Anchors, Axis Sizes
    return anchors, axis_1, axis_2

# Function to calculate anchor stride
def calculate_anchor_stride(original_size, feature_map_size):
    """
    Calculates the stride based on the original size and feature map size.

    Parameters
    ----------
    original_size: int
        Size of the original image (width or height).
    feature_map_size: int
        Size of the feature map (width or height).

    Returns
    ----------
    int
        Stride value.
    """
    return original_size // feature_map_size

# Function to create anchor centers
def create_anchor_centers(feature_map_width, feature_map_height, image_width=800, image_height=800):
    """
    Creates anchor centers for the feature map.

    Parameters
    ----------
    feature_map_width: int
        Width of the feature map.
    feature_map_height: int
        Height of the feature map.
    image_width: int
        Width of the original image (default is 800).
    image_height: int
        Height of the original image (default is 800).

    Returns
    ----------
    np.ndarray
        Array of anchor center coordinates.
    """
    # Calculate strides
    anchor_stride_x = calculate_anchor_stride(image_width, feature_map_width)
    anchor_stride_y = calculate_anchor_stride(image_height, feature_map_height)

    # Creating anchor centers based on the feature map size, 0.5 to center for each pixel
    anchors_x_center = (np.arange(feature_map_width) + 0.5)
    anchors_y_center = (np.arange(feature_map_height) + 0.5)

    # Create a grid of anchor center coordinates based on the feature map size
    anchor_centers = np.array(np.meshgrid(anchors_x_center, anchors_y_center, indexing='xy')).T.reshape(-1, 2)

    return anchor_centers

# Create aspect ratio and scale anchor boxes
def create_aspect_boxes(anchor_box_ratios, anchor_box_scales, base_anchor_box):
    """
    Creates anchor boxes of different ratios and scales

    Parameters
    ----------
    anchor_box_ratio: Different aspect ratios used for the bounding boxes
    anchor_box_scales: Different scale ratios used for the bounding boxes
    base_anchor_box: Base anchor box size
    ----------

    Returns
    ----------
    aspect_box
        List of anchor boxes with different scales and aspect ratios for each anchor point
    ----------
    """

    aspect_box = []
    # Iterate Over the ratios and scales and transform the bounding box dimensions.
    for ratios in anchor_box_ratios:
        for scales in anchor_box_scales:
            if ratios < 1:
                # Taller Box
                width = base_anchor_box[0] * scales * ratios
                height = base_anchor_box[1] * scales
            else:
                # Wider Box
                width = base_anchor_box[0] * scales
                height = base_anchor_box[1] * scales / ratios
            aspect_box.append([width,height])
    return aspect_box

# Creating Relative Coordinates To the center of the Anchor Box
def relative_coordinates(aspect_boxes, x_c, y_c):
    """
    Creates a grid of the anchor points for the image

    Parameters
    ----------
    aspect_boxes: Array of bounding boxes with different aspect ratios and scales
    x_c: X center of the anchor
    y_c: Y center of the anchor
    ----------

    Returns
    ----------
    final_box
        List of anchor boxes for each anchor point
    ----------
    """
    final_box = []
    # Iterate over the transformed boxes and create coordinate relative to the center pixel.
    for boxes in aspect_boxes:
        width_alignment = boxes[0] / 2
        height_alignment = boxes[1] / 2
        final_box.append([x_c - width_alignment, y_c - height_alignment, x_c + width_alignment, y_c + height_alignment])
    # Array of bounding boxes relative to the pixel
    return final_box

# Function to initialize all anchor boxes for the Faster R-CNN
def initialize_all_anchor_boxes(original_img_shape, feature_map_shape):
    """
    Creates the anchor boxes for feature map that can be used by the model

    Parameters
    ----------
    original_img_shape: Shape of the original image
    feature_map_shape: Shape of the feature map, without the batch dimension
    ----------

    Returns
    ----------
    feature_map_anchor_box_coordinates
        Tensor of anchor boxes for the feature map (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_OF_ANCHORS_PER_PIXEL,4)
    ----------
    """
    # Calculate the anchor center strides for the X & Y axes
    anchor_stride_x , anchor_stride_y = calculate_anchor_stride(original_img_shape[1], feature_map_shape[1]) , calculate_anchor_stride(original_img_shape[2], feature_map_shape[2])
    # Calculating the anchor centers for the x and y axes
    anchor_centers = create_anchor_centers(feature_map_shape[1], feature_map_shape[2])
    # Calculating the different aspect ratios and scales for the anchor boxes
    aspect_ratios = tf.constant(create_aspect_boxes(ANCHOR_BOX_RATIOS,ANCHOR_BOX_SCALES,BASE_ANCHOR_BOX_SIZE),dtype=tf.float32)
    # Scaling the anchor box ratios to the feature map size since they are in the original image space for now
    feature_map_aspect_ratios = tf.stack([aspect_ratios[:,0]/anchor_stride_x,aspect_ratios[:,1]/anchor_stride_y],axis=1)
    
    # Calculating the anchor boxes for each center
    # Calculating the X,Y offsets to be added to the center point
    offset_x,offset_y = tf.cast(feature_map_aspect_ratios[:,0]/2,dtype=tf.float32), tf.cast(feature_map_aspect_ratios[:,1]/2,tf.float32)

    # Calculate the values by adding the offsets to the centers separately
    x_centers,y_centers = anchor_centers[:,0] , anchor_centers[:,1]
    # Repeat the tensor so that it is reshaped for the total number of anchor boxes per pixel since the centers for each of the aspect ratio boxes are the same
    x_centers = tf.cast(tf.repeat(x_centers,repeats=tf.shape(feature_map_aspect_ratios)[0]),tf.float32)
    y_centers = tf.cast(tf.repeat(y_centers,repeats=tf.shape(feature_map_aspect_ratios)[0]),tf.float32)

    # Tile the tensor so that the offsets are tiled since the aspect ratios are iterated over
    offset_x,offset_y = tf.tile(offset_x,[tf.shape(anchor_centers)[0]]),tf.tile(offset_y,[tf.shape(anchor_centers)[0]])
    
    # Calculate the four coordinates for the anchor box using the centers
    x_1 = x_centers - offset_x
    x_2 = x_centers + offset_x
    y_1 = y_centers - offset_y
    y_2 = y_centers + offset_y

    # Clipping anchor boxes to stop the the coordinates from going out of bounds
    x_1 = tf.maximum(x_1,0)
    x_2 = tf.minimum(x_2,feature_map_shape[1])
    y_1 = tf.maximum(y_1,0)
    y_2 = tf.minimum(y_2,feature_map_shape[2])
    
    # Stack the coordinates together in format of (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,4)
    feature_map_anchor_box_coordinates = tf.stack([x_1,y_1,x_2,y_2],axis=1)
    feature_map_anchor_box_coordinates = tf.reshape(feature_map_anchor_box_coordinates,[feature_map_shape[1],feature_map_shape[2],-1,4])
    
    # Adding a batch dimension so that it can be tiled
    feature_map_anchor_box_coordinates = tf.expand_dims(feature_map_anchor_box_coordinates,axis=0)
    # Tiling the vector on the batch dimension only, everything else is kept as it is.
    feature_map_anchor_box_coordinates = tf.tile(feature_map_anchor_box_coordinates,[feature_map_shape[0],1,1,1,1])
    
    
    return feature_map_anchor_box_coordinates

def IOU_scores(ground_truth_boxes,predicted_boxes):
    """
    Calculate IOU Scores For Ground Truth Boxes & All Prediction Boxes

    Parameters:
    ---------
    ground_truth_boxes : Tensor
        Tensor Of Ground Truth Boxes (B,NO_OF_GT_BOXES,4)

    predicted_boxes : Tensor
        Tensor Of Prediction Boxes (B,FEATURE_MAP_WIDTH * FEATURE_MAP_HEIGHT * NUM_OF_ANCHORS_PER_PIXEL,4)    

    Returns:
    -------
    iou_scores: Tensor
        Tensor Of IOU Scores (B,NO_OF_GT_BOXES,NO_OF_PRED_BOXES)
        NOTE: Each row is a ground truth box and each column is a anchor prediction box
    """
    
    # First we need to split up the two boxes into four coordinates each
    x11,y11,x21,y21 = tf.split(ground_truth_boxes,num_or_size_splits=4,axis=-1)
    x12,y12,x22,y22 = tf.split(predicted_boxes,num_or_size_splits=4,axis=-1)

    x11 = tf.expand_dims(x11,axis=2)
    y11 = tf.expand_dims(y11,axis=2)
    x21 = tf.expand_dims(x21,axis=2)
    y21 = tf.expand_dims(y21,axis=2)

    x12 = tf.expand_dims(x12,axis=1)
    y12 = tf.expand_dims(y12,axis=1)
    x22 = tf.expand_dims(x22,axis=1)
    y22 = tf.expand_dims(y22,axis=1)
    
    # Now we to get the max and min values for each axes for each top left and bottom right corner of the box
    x1_max = tf.math.maximum(x11,x12)
    y1_max = tf.math.maximum(y11,y12)

    x2_min = tf.math.minimum(x21,x22)
    y2_min = tf.math.minimum(y21,y22)

    # Now we need to calculate the width and height of the intersection box, we need to stop it from being non-negative as well
    width = tf.math.maximum(0.0,x2_min - x1_max)
    height = tf.math.maximum(0.0,y2_min - y1_max)

    # Calculate the area of the intersection box for IOU scores
    area_of_intersection = width * height

    # Calculate area of ground truth boxes
    ground_truth_boxes_area = (x21-x11) * (y21-y11)

    # Calculate are of predicted boxes
    predicted_boxes_area = (x22 - x12) * (y22-y12)
    
    # Calculate the are of the union of the two boxes using venn diagram formula
    area_of_union = tf.maximum(0.0,ground_truth_boxes_area + predicted_boxes_area - area_of_intersection)

    # Calculate IOU score using the formula
    iou_scores = area_of_intersection / (area_of_union + 1e-6)
   
    return tf.squeeze(iou_scores, axis=-1)

# This function assigns labels to the anchor boxes based on the IOU scores with the ground truth
def assign_object_label(iou_scores_tensor,IOU_FOREGROUND_THRESH = RPN_POSITIVE_ANCHOR_IOU_THRESHOLD,IOU_BACKGROUND_THRESH = RPN_NEGATIVE_ANCHOR_IOU_THRESHOLD,FEATURE_MAP_WIDTH= 50,FEATURE_MAP_HEIGHT= 50,NUM_OF_ANCHORS_PER_PIXEL= NUM_OF_ANCHORS_PER_PIXEL):
    """
    Assign Object Labels If They Are Foreground (Object) or Background

    Parameters:
    ---------
    iou_scores_tensor : Tensor
        Tensor Of IOU Scores Shape (B,GT_BOXES,PRED_BOXES)

    Returns:
    -------
    object_label_tensor: Tensor
        Tensor Of Object Labels (B, FEATURE_MAP_WIDTH, FEATURE_MAP_HEIGHT, NUM_OF_ANCHORS_PER_PIXEL, 1)
    """
    
    # Need to implement the rule where max IOU per anchor box is considered
    max_iou_per_anchor_box = tf.reduce_max(iou_scores_tensor,axis=1)
    
    # Everything is considered a background. So it does not need to check for negative labels.
    object_label_tensor = tf.zeros_like(max_iou_per_anchor_box,dtype=tf.int64)

    # Calculate Positive Labels -> Ignored Labels
    object_label_tensor = tf.where(max_iou_per_anchor_box >= IOU_FOREGROUND_THRESH,1,object_label_tensor)
    object_label_tensor = tf.where((max_iou_per_anchor_box <= IOU_FOREGROUND_THRESH) & (max_iou_per_anchor_box >= IOU_BACKGROUND_THRESH),-1,object_label_tensor)

    # Need to consider a proposal for each anchor box so the max per ground truth box
    max_iou_indices_per_gt_box = tf.argmax(iou_scores_tensor,axis=-1) # Looks at the anchors in the last axis
    batch_range = tf.range(max_iou_indices_per_gt_box.shape[0],dtype=tf.int64) # Create a range for that can be repeated for the max IOU per GT Box
    batch_rank = tf.repeat(batch_range,[max_iou_indices_per_gt_box.shape[1]]) # Each image will have fixed NUM_GT_BOXES
    flattened_max_iou_indices = tf.reshape(max_iou_indices_per_gt_box,[-1]) # Flattened the indices to make it easier for stacking
    stacked_ranks = tf.stack([batch_rank,flattened_max_iou_indices],axis=-1) # Stacked the ranks to make it easier to update the main index
    object_label_tensor = tf.tensor_scatter_nd_update(object_label_tensor,stacked_ranks,tf.ones_like(flattened_max_iou_indices))
    
    # Reshaping the objectness labels to the shape (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_OF_ANCHORS_PER_PIXEL,1)
    object_label_tensor = tf.reshape(object_label_tensor,(iou_scores_tensor.shape[0],FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_OF_ANCHORS_PER_PIXEL,1))

    return object_label_tensor

# Function that will generate the objectness labels (0-background,1-foreground,-1-Ignore) for the anchor boxes on the feature map
def generate_objectness_labels(anchor_boxes,gt_boxes,anchor_stride_x = 16,anchor_stride_y = 16):
    """
    Generate objectness labels for the anchor boxes in the feature map
     0 - Background
     1 - Object/Foreground
    -1 - Ignore Object

    Parameters:
    ---------
    anchor_boxes : Tensor
        Tensor of anchor boxes for the feature map (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_OF_ANCHORS_PER_PIXEL,4)
    
    gt_boxes : Tensor
        Tensor Of Ground Truth Boxes (B,NO_OF_GT_BOXES,4)

    Returns:
    -------
    object_labels : Tensor
        Tensor of object labels based on the IOU Thresholds (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_OF_ANCHORS_PER_PIXEL,1)
    iou_matrix : Tensor
        Tensor of IOU scores that are calculated for the ground truth boxes and the anchor boxes (B,GT_BOXES,NUM_TOTAL_ANCHORS)
    """
    batch_size , feature_map_width, feature_map_height , num_of_anchors_per_pixel = anchor_boxes.shape[0],anchor_boxes.shape[1],anchor_boxes.shape[2],anchor_boxes.shape[3]

    # Get IOU Matrix For the anchor boxes
    # Need to scale the anchor boxes from the feature space to the image space.
    anchor_boxes = anchor_boxes * anchor_stride_x
    # Flattened the anchor boxes to [B,FEATURE_MAP_WIDTH * FEATURE_MAP_HEIGHT * NUM_OF_ANCHORS_PER_PIXEL,4]
    anchor_boxes =  tf.reshape(anchor_boxes,[anchor_boxes.shape[0],anchor_boxes.shape[1]*anchor_boxes.shape[2]*anchor_boxes.shape[3],4])
    
    iou_matrix = IOU_scores(gt_boxes,anchor_boxes) # IOU matrix for the anchor boxes and the ground truth boxes
    # Reshaping the IOU matrix to be in the shape of the original anchor boxes
    # iou_matrix = tf.reshape(iou_matrix,[batch_size,feature_map_width,feature_map_height,1])

    # Assigning labels for the IOU matrix based on the thresholding
    object_labels = assign_object_label(iou_matrix)

    return object_labels,iou_matrix

def create_proposals_from_rpn(anchors,deltas,image_stride = 16):
    """
    Create the proposals using the anchors and adding predicted deltas from the RPN

    Parameters:
    ---------
    anchors: Tensor
        Anchors in the shape of (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_ANCHORS,4)

    deltas: Tensor
        Deltas in the shape of (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_ANCHORS,4)

    Returns:
    -------

    proposals: Tensor
        Proposals in the shape of (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_ANCHORS,4)
    """
    # Create the proposals by adding the deltas to the anchor boxes
    proposals = []
    for image in range(anchors.shape[0]):
        # Get the image anchors and then calculate the center format to add the deltas
        image_anchors = convert_xy_boxes_to_center_format(anchors[image] * image_stride)
        image_deltas = deltas[image]
        # Applying the bounding box deltas to the anchors
        image_proposals = apply_bounding_box_deltas(image_anchors,image_deltas)

        proposals.append(image_proposals)

    # Stacking the proposals to create one batch
    proposals = tf.stack(proposals,axis=0)

    return proposals

# Filter by minimum size to clip the proposals to the right shape to remove degenerate boxes
def filter_proposals_by_size(proposals,scores,min_size = 1):
    """
    Filter proposals by the size to remove the small boxes which can cause some effect on the errors of the model

    Parameters:
    ---------
    proposals: Tensor
        Proposals in the shape of (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_ANCHORS,4)

    scores: Tensor
        Scores in the shape of (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_ANCHORS,1)

    Returns:
    -------

    stacked_filtered_proposals: Tensor
        Proposals in the shape of (B,NUM_FILTERED_ANCHORS,4)

    stacked_filtered_scores: Tensor
        Scores in the shape of (B,NUM_FILTERED_ANCHORS,1)
    """
    filtered_proposals = []
    filtered_scores = []
    for image in range(proposals.shape[0]):
        x_min,y_min,x_max,y_max = tf.split(proposals[image], num_or_size_splits = 4,axis = -1)

        # Calculate the width and height
        height = y_max - y_min
        width = x_max - x_min

        # Use logical AND to get the indices that hold up this condition
        valid_indices = tf.where(tf.math.logical_and(tf.squeeze(width) >= min_size,tf.squeeze(height) >= min_size))

        # Gathering the proposals that are valid
        valid_proposals = tf.gather_nd(proposals[image],valid_indices)

        # Gathering the corresponding scores that are valid
        valid_scores = tf.gather_nd(scores[image],valid_indices)

        filtered_proposals.append(valid_proposals)
        filtered_scores.append(valid_scores)

    # Padding the two images
    num_of_max_proposals = tf.reduce_max([tf.shape(dim)[0] for dim in filtered_proposals])
    padded_tensors = [tf.pad(tensor, paddings = [[0,num_of_max_proposals - tf.shape(tensor)[0]],[0,0]]) for tensor in filtered_proposals]
    padded_score_tensors = [tf.pad(tensor, paddings = [[0,num_of_max_proposals - tf.shape(tensor)[0]]],constant_values=0.0) for tensor in filtered_scores]

    stacked_filtered_proposals = tf.stack(padded_tensors,axis=0)
    stacked_filtered_scores = tf.stack(padded_score_tensors,axis=0)

    return stacked_filtered_proposals, stacked_filtered_scores
    
# Pre-NMS Top K
# There is an edge case if the proposals are less than the thresh but since the value is less than the usual so that is not needed
def pre_nms_top_k(proposals,scores,k_thresh = 12000):
    """
    Perform Top K Selection before NMS to the threshold set for the function to remove bad boxes

    Parameters:
    ---------
    proposals: Tensor
        Proposals in the shape of (B,NUM_FILTERED_ANCHORS,4)

    scores: Tensor
        Scores in the shape of (B,NUM_FILTERED_ANCHORS,1)

    Returns:
    -------

    stacked_filtered_proposals: Tensor
        Proposals in the shape of (B,NUM_FILTERED_ANCHORS,4)

    stacked_filtered_scores: Tensor
        Scores in the shape of (B,NUM_FILTERED_ANCHORS,1)
    """
    
    # Putting a self guard for K
    if k_thresh > proposals.shape[1]:
        tf.print(f"WARNING: K Threshold {k_thresh} pre-NMS is greater than the number of proposals {proposals.shape[1]}, setting K to {proposals.shape[1]}")
        pass
    k_thresh = tf.minimum(k_thresh,proposals.shape[1])
    
    top_k_proposals = []
    top_k_scores = []
    for image in range(proposals.shape[0]):
        flattened_scores = tf.reshape(scores[image],[-1])
          
        # Getting the top k scores and indices
        top_scores,top_indices = tf.math.top_k(flattened_scores,k=k_thresh)

        # Getting the proposals using the indices
        flattened_proposals = tf.reshape(proposals[image],[-1,4])
        top_proposals = tf.gather(flattened_proposals,top_indices)
        
        top_k_proposals.append(top_proposals)
        top_k_scores.append(top_scores)

    # Stack them together
    stacked_top_proposals = tf.stack(top_k_proposals,axis=0)
    stacked_top_scores = tf.stack(top_k_scores,axis=0)

    return stacked_top_proposals, stacked_top_scores

# NMS for Proposals
def class_agnostic_nms_thresholding(proposals,scores,max_output_size=5000,iou_thresholding = 0.85):
    """
    Perform Class agnostic NMS thresholding for the proposals to reduce overlapped functions
    NOTE: There is a padding feature in this function that can get triggered
    
    Parameters:
    ---------
    proposals: Tensor
        Proposals in the shape of (B,TOP_K,4)

    scores: Tensor
        Scores in the shape of (B,TOP_K,1)

    Returns:
    -------

    stacked_proposals: Tensor
        Proposals in the shape of (B,NMS_OUTPUT,4)

    stacked_filtered_scores: Tensor
        Scores in the shape of (B,NMS_OUTPUT,1)
    """
    nms_proposals = []
    nms_scores = []
    for image in range(proposals.shape[0]):
        
        x_min,y_min,x_max,y_max = tf.split(proposals[image], num_or_size_splits = 4,axis = -1)
        
        proposals_stacked = tf.concat([y_min,x_min,y_max,x_max] ,axis=-1)
        
        selected_indices = tf.image.non_max_suppression(proposals_stacked,scores[image],max_output_size=max_output_size,iou_threshold = iou_thresholding)

        selected_proposals = tf.gather(proposals[image],selected_indices)

        selected_scores = tf.gather(scores[image],selected_indices)

        nms_proposals.append(selected_proposals)
        nms_scores.append(selected_scores)

    num_of_max_proposals = tf.reduce_max([tf.shape(dim)[0] for dim in nms_proposals])
    dummy_box = tf.constant([[0,0,0,0]],dtype=nms_proposals[0].dtype)
    padded_tensors = [tf.concat([tensor,tf.tile(dummy_box,multiples=[num_of_max_proposals - tf.shape(tensor)[0],1])],axis=0) for tensor in nms_proposals]
    padded_score_tensors = [tf.pad(tensor, paddings = [[0,num_of_max_proposals - tf.shape(tensor)[0]]],constant_values=-1e-9) for tensor in nms_scores]
    
    stacked_proposals = tf.stack(padded_tensors,axis = 0)
    stacked_scores = tf.stack(padded_score_tensors,axis=0)

    return stacked_proposals, stacked_scores

# Post NMS Top K
def post_nms_top_k(proposals,scores,k_thresh = 2000):
    """
    Perform Top K Post NMS thresholding to keep the top proposals
    
    Parameters:
    ---------
    proposals: Tensor
        Proposals in the shape of (B,TOP_K,4)

    scores: Tensor
        Scores in the shape of (B,TOP_K,1)

    Returns:
    -------

    stacked_top_proposals: Tensor
        Proposals in the shape of (B,TOP_K,4)

    stacked_filtered_scores: Tensor
        Scores in the shape of (B,TOP_K,1)
    """
    
    # Putting a self guard for K
    if k_thresh > proposals.shape[1]:
        tf.print(f"WARNING: K Threshold {k_thresh} post-NMS is greater than the number of proposals {proposals.shape[1]}, setting K to {proposals.shape[1]}")
        
    k_thresh = tf.minimum(k_thresh,proposals.shape[1])
    
    top_k_proposals = []
    top_k_scores = []
    for image in range(proposals.shape[0]):
          
        # Getting the top k scores and indices
        top_scores,top_indices = tf.math.top_k(scores[image],k=k_thresh)

        # Getting the proposals using the indices
        flattened_proposals = tf.reshape(proposals[image],[-1,4])
        top_proposals = tf.gather(flattened_proposals,top_indices)
        
        top_k_proposals.append(top_proposals)
        top_k_scores.append(top_scores)

    # Stack them together
    stacked_top_proposals = tf.stack(top_k_proposals,axis=0)
    stacked_top_scores = tf.stack(top_k_scores,axis=0)

    return stacked_top_proposals, stacked_top_scores

# Assign the objectness labels for the proposals
def assign_proposal_object_labels(iou_scores_tensor,iou_score_threshold = 0.4):
    """
    Assign Object Labels If They Are Foreground (Object) or Background, No ignore label for the RoI Pipeline

    Parameters:
    ---------
    iou_scores_tensor : Tensor
        Tensor Of IOU Scores Shape (B,GT_BOXES,PRED_BOXES)

    Returns:
    -------
    object_label_tensor: Tensor
        Tensor Of Object Labels (B, PRED_BOXES, 1)
    """
    
    # Need to implement the rule where max IOU per anchor box is considered
    max_iou_per_anchor_box = tf.reduce_max(iou_scores_tensor,axis=1)
    
    # Everything is considered a background. So it does not need to check for negative labels.
    object_label_tensor = tf.zeros_like(max_iou_per_anchor_box,dtype=tf.int64)

    # Calculate Positive Labels -> Ignored Labels
    object_label_tensor = tf.where(max_iou_per_anchor_box > iou_score_threshold,1,object_label_tensor)

    # Need to consider a proposal for each anchor box so the max per ground truth box
    max_iou_indices_per_gt_box = tf.argmax(iou_scores_tensor,axis=-1) # Looks at the anchors in the last axis
    batch_range = tf.range(max_iou_indices_per_gt_box.shape[0],dtype=tf.int64) # Create a range for that can be repeated for the max IOU per GT Box
    batch_rank = tf.repeat(batch_range,[max_iou_indices_per_gt_box.shape[1]]) # Each image will have fixed NUM_GT_BOXES
    flattened_max_iou_indices = tf.reshape(max_iou_indices_per_gt_box,[-1]) # Flattened the indices to make it easier for stacking
    stacked_ranks = tf.stack([batch_rank,flattened_max_iou_indices],axis=-1) # Stacked the ranks to make it easier to update the main index
    object_label_tensor = tf.tensor_scatter_nd_update(object_label_tensor,stacked_ranks,tf.ones_like(flattened_max_iou_indices))

    return object_label_tensor

# Sample 512 boxes per image
def sample_proposals_per_image(proposals,scores,ground_truth_boxes,sample_size_per_image = 512,positive_composition_ratio = 0.25,seed = None):
    """
    Assign Object Labels If They Are Foreground (Object) or Background, No ignore label for the RoI Pipeline

    Parameters:
    ---------
    proposals : Tensor
        Tensor Of IOU Scores Shape (B,TOP_K,4)

    scores : Tensor
        Tensor Of IOU Scores Shape (B,TOP_K,1)

    ground_truth_boxes: Tensor
        Tensor of the ground truth boxes (B,GT_BOXES,4)

    Returns:
    -------
    sampled_proposals: Tensor
        Tensor Of Object Labels (B, SAMPLED_BOXES, 4)

    sampled_proposals: Tensor
        Tensor Of Object Labels (B, SAMPLED_BOXES, 1)
    """
    # Create the roi labels first
    proposal_iou = IOU_scores(ground_truth_boxes,proposals)
    proposal_labels = assign_proposal_object_labels(proposal_iou)
    
    # Making sure that the sample size is not greater than the number of proposals
    if sample_size_per_image > proposals.shape[1]:
        tf.print(f"WARNING: Sample Size Per Image {sample_size_per_image} is greater than the number of proposals {proposals.shape[1]}, setting Sample Size to {proposals.shape[1]}")
        
    sample_size_per_image = tf.cast(sample_size_per_image, tf.int32)  
    sample_size_per_image = tf.minimum(sample_size_per_image,tf.shape(proposals)[1])

    # Calculate the sampling splits
    max_positives = tf.cast(tf.round(tf.cast(sample_size_per_image,tf.float32) * tf.constant(positive_composition_ratio, tf.float32)),tf.int32)
    B = proposal_labels.shape[0]
    sampled_proposals = []
    sampled_scores = []
    
    for image in range(B):
        
        positive_indices = tf.where(proposal_labels[image] == 1)
        negative_indices = tf.where(proposal_labels[image] == 0)

        num_positive_anchors = tf.minimum(tf.shape(positive_indices)[0],max_positives)
        num_negative_anchors = tf.minimum(tf.shape(negative_indices)[0],sample_size_per_image - num_positive_anchors)
        
        # Checking for defecit using negative anchors
        defecit = sample_size_per_image - (num_positive_anchors + num_negative_anchors)
        negative_room = tf.maximum(0,tf.shape(negative_indices)[0] - num_negative_anchors)
        extra_negatives = tf.minimum(defecit,negative_room)
        num_negative_anchors = num_negative_anchors + extra_negatives
        
        # Check for defecit using positive anchors
        defecit = sample_size_per_image - (num_positive_anchors + num_negative_anchors)
        positive_room = tf.maximum(0,tf.shape(positive_indices)[0] - num_positive_anchors)
        extra_positives = tf.minimum(defecit,positive_room)
        num_positive_anchors = num_positive_anchors + extra_positives

        positive_indices = tf.random.shuffle(positive_indices,seed=seed)[:num_positive_anchors]
        negative_indices = tf.random.shuffle(negative_indices,seed=seed)[:num_negative_anchors]

        positive_anchors = tf.gather_nd(proposals[image],positive_indices)
        negative_anchors = tf.gather_nd(proposals[image],negative_indices)
        
        sampled_anchors = tf.concat([positive_anchors,negative_anchors],axis=0)
        
        sampled_proposals.append(sampled_anchors)

        positive_scores = tf.gather_nd(scores[image],positive_indices)
        negative_scores = tf.gather_nd(scores[image],negative_indices)

        sampled_nms_scores = tf.concat([positive_scores,negative_scores],axis=0)

        sampled_scores.append(sampled_nms_scores)

    sampled_proposals = tf.stack(sampled_proposals,axis=0)
    sampled_scores = tf.stack(sampled_scores,axis=0)
    
    return sampled_proposals, sampled_scores       

# Creating a function for getting the positive anchor boxes
def get_positive_anchor_boxes_and_corresponding_offsets(anchor_boxes,offsets,object_labels):
    """
    Get the positive anchor boxes based on their object labels
    NOTE: The output anchor boxes are in feature map coordiantes
    Parameters:
    ---------
    anchor_boxes : Tensor
        Tensor of the anchor boxes in the shape (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_OF_ANCHORS_PER_PIXEL,4)

    offsets : Tensor
        Tensor of the offsets in the shape (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_OF_ANCHORS_PER_PIXEL,4)
        
    object_labels: Tensor
        Tensor of the object labels

    Returns:
    -------
    positive_anchor_boxes:
        Positive anchor boxes in the shape of (B,NUM_POS_ANCHORS,4) in the XY Coordinate system
        
    offsets_for_positive_anchor_boxes:
        Offsets for the positive anchor boxes in the shape(B,NUM_POS_ANCHORS , 4) in the Xc,Yc,W,H system
    """
    # Creating the positive mask and then getting the indices
    # We need to gather the bounding boxes
    positive_mask = tf.squeeze(object_labels,axis=-1) == 1 # Squeezing the last axis from the object labels
    positive_indices = tf.where(positive_mask) # Shape of (NUM_POS_ANCHOR_BOXES,4) [B, H, W, BOX_NO]

    # Need to calculate them per image in the batch not batchwise
    batch_indices = positive_indices[:,0] # Getting the batch index from the positive anchors
    batch_size = tf.reduce_max(batch_indices) + 1 # Zero-indexing for batches

    # Getting the positive anchors
    positive_anchors_indices = []
    positive_anchors = []
    positive_offsets = []
    for index in range(batch_size):
        # Create a batch mask for the indices to select one batch at a time
        batch_mask = tf.equal(batch_indices,index)
        # Getting positive anchor boxes indices in the batch
        positive_anchor_boxes_in_batch = tf.boolean_mask(positive_indices,batch_mask)
        
        positive_anchors.append(tf.gather_nd(anchor_boxes,positive_anchor_boxes_in_batch))
        positive_offsets.append(tf.gather_nd(offsets,positive_anchor_boxes_in_batch))
        positive_anchors_indices.append(positive_anchor_boxes_in_batch)
        
    # Padding the batches
    max_pos_anchors = max(tensor.shape[0] for tensor in positive_anchors)
    padded_anchors = []
    # Iterating over the batches and padding them
    for batch in positive_anchors:
        # Padding the anchors
        padding = [[0,max_pos_anchors-batch.shape[0]],[0,0]]
        padded_tensor = tf.pad(batch,padding,constant_values = 0)
        padded_anchors.append(padded_tensor)
    
    padded_offsets = []
    for batch in positive_offsets:
        # Padding the anchors
        padding = [[0,max_pos_anchors-batch.shape[0]],[0,0]]
        padded_tensor = tf.pad(batch,padding,constant_values = 0)
        padded_offsets.append(padded_tensor)

    padded_indices = []
    for batch in positive_anchors_indices:
        # Padding the anchors
        padding = [[0,max_pos_anchors-batch.shape[0]],[0,0]]
        padded_tensor = tf.pad(batch,padding,constant_values = -1)
        padded_indices.append(padded_tensor)

    # Stacking the padded tensors together
    positive_anchors = tf.stack(padded_anchors)
    offsets_for_positive_anchor_boxes = tf.stack(padded_offsets)
    positive_anchors_indices = tf.stack(padded_indices)

    return positive_anchors, offsets_for_positive_anchor_boxes, positive_anchors_indices

def rpn_sample_anchors(labels,batch_sample_per_image = 256,positive_composition_ratio = 0.5, seed = None):
    """
    Function to Sample the RPN anchors to stabilize the errors and stopping one side from overpowering the other
     
    Parameters:
    ---------
    labels: Tensor
        Object labels that are in the shape of the (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_ANCHORS,1) this is either [-1,0,1]
        
    batch_sample_per_image : Int
        Sample Size for each image that is used to limit the sampling size that can be changed if the errors are too large
        
    positive_composition_ratio: Float
        Positive composition ratio, default value is 50% but it works with up to the limit if the positives are low, the values needs to be changed
        
    Returns:
    -------
    sampled_labels : Tensor
        Sampled object labels that are in the shape of the (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_ANCHORS,1) this is either [-1,0,1]
    """
    
    labels = tf.squeeze(labels)
    
    # Get the ratio for the sampling split
    max_positives = round(batch_sample_per_image * positive_composition_ratio)

    # Getting the samples per image
    B = labels.shape[0]
    sampled_labels = []
    for image in range(B):
        # Getting the positive and negative indices per image
        positive_indices = tf.where(labels[image] == 1)
        negative_indices = tf.where(labels[image] == 0)

        # Getting the split of the image
        # This will handle when the positives are less that 50% too but the scales might need to be shifted if value too far off too often
        num_positive_anchors = tf.minimum(tf.shape(positive_indices)[0],max_positives)
        num_negative_anchors = tf.minimum(tf.shape(negative_indices)[0],batch_sample_per_image - num_positive_anchors)

        # Random shuffling
        positive_indices = tf.random.shuffle(positive_indices,seed=seed)[:num_positive_anchors]
        negative_indices = tf.random.shuffle(negative_indices,seed=seed)[:num_negative_anchors]

        # Creating the tensor for the image
        H,W,A = labels[image].shape
        object_labels = tf.fill([H,W,A],tf.constant(-1,labels.dtype)) # Starting with the ignored anchors

        # Updating the labels with the sampled positive indices
        object_labels = tf.tensor_scatter_nd_update(object_labels,positive_indices,tf.ones([num_positive_anchors],labels.dtype))
        object_labels = tf.tensor_scatter_nd_update(object_labels,negative_indices,tf.zeros([num_negative_anchors],labels.dtype))

        sampled_labels.append(object_labels)

    sampled_labels = tf.stack(sampled_labels,axis = 0)
    
    return tf.expand_dims(sampled_labels,axis=-1)

# Implementing the function for Binary log loss between the object labels and the predicted values
def calculate_objectness_loss(predicted_scores,target_object_labels):
    """
    Calculate the binary cross entropy loss for the model between the objectness predicted score from the RPN and the target labels based on the IOU matrix

    Parameters:
    ---------
    predicted_scores : Tensor
        Tensor of the foreground scores of the predicted scores in the shape (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_OF_ANCHORS_PER_PIXEL,1)
        
    target_object_labels: Tensor
        Tensor of the target object labels which are obtained from the IOU matrix and the IOU Threshold rules

    Returns:
    -------
    objectness_loss:
        Binary Cross Entropy Loss for the objectness model scores
    """

    # We have the object (1), background (0) and ignored labels (-1).
    # We have to mask the values in the tensor so that unnecessary information is not taken into account.
    # boolean_mask_matrix = target_object_labels != -1
    # mask = tf.cast(boolean_mask_matrix,dtype=tf.int32)
    # retained_labels = target_object_labels * mask
    # retained_labels = tf.where(target_object_labels == -1, tf.zeros_like(target_object_labels), target_object_labels)
    # binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()
    # objectness_loss = binary_cross_entropy(retained_labels,predicted_scores)

    pred_scores = tf.reshape(predicted_scores,[-1])
    object_labels = tf.reshape(target_object_labels,[-1])

    # Filter out the ones where its -1 to stop the loss function from penalizing the ignored boxes
    # by forcing them into background
    valid_mask = tf.not_equal(object_labels,-1)
    pos_pred_scores = tf.boolean_mask(pred_scores,valid_mask)
    pos_target_labels = tf.boolean_mask(object_labels,valid_mask)
    pos_target_labels = tf.cast(pos_target_labels,dtype=tf.float32)
    
    loss_fn = BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    # Calculate the loss now with the valid scores
    objectness_loss = tf.reduce_mean(loss_fn(pos_target_labels,pos_pred_scores))

    return objectness_loss

# Creating a function to convert bounding boxes from (X_MIN,Y_MIN,X_MAX,Y_MAX) to (Xc,Yc,W,H)
def convert_xy_boxes_to_center_format(anchor_boxes):
    """
    Converts the boxes from the standard format of (X_MIN,Y_MIN,X_MAX,Y_MAX) used by me into the format
    (Xc,Yc,W,H) used by the model

    Parameters:
    ---------
    anchor_boxes: Tensor
        Tensor of anchor boxes in the format of (X_MIN,Y_MIN,X_MAX,Y_MAX)

    Returns:
    -------
    converted_boxes: Tensor
        Tensor of anchor boxes in the converted format of (Xc,Yc,W,H)    
    """
    # Creating a utility function for easier conversion in the model since all the anchor boxes made by me are in (X_MIN,Y_MIN,X_MAX,Y_MAX)
    # the model predicts the bounding boxes, its offsets, its errors in the shape of (Xc,Yc,W,H)

    # Split the anchor boxes to be get the individual coordinates
    x_min,y_min,x_max,y_max = tf.split(anchor_boxes,num_or_size_splits = 4, axis = -1)

    # Calculating the conversion to the center format
    x_c = (x_max + x_min) / 2  # Center X for the box using midpoint formula
    y_c = (y_max + y_min) / 2  # Center Y for the box using the midpoint formula
    w = (x_max - x_min) # Width of the box
    h = (y_max - y_min) # Height of the box

    converted_boxes = tf.concat([x_c,y_c,w,h],axis=-1) # Stacking the boxes together into one vector
    
    return converted_boxes

# Creating a function to convert the bounding boxes from (Xc,Yc,W,H) to (X_MIN,Y_MIN,X_MAX,Y_MAX)
def convert_center_format_boxes_to_xy_coordinate(anchor_boxes):
    """
    Converts the boxes from the center format of (Xc,Yc,W,H) used by me into the standard format
    (X_MIN,Y_MIN,X_MAX,Y_MAX) used by the model

    Parameters:
    ---------
    anchor_boxes: Tensor
        Tensor of anchor boxes in the format of (Xc,Yc,W,H) 

    Returns:
    -------
    converted_boxes: Tensor
        Tensor of anchor boxes in the converted format of (X_MIN,Y_MIN,X_MAX,Y_MAX)   
    """
    # Creating a utility function for easier conversion in the model since all the anchor boxes made by me are in (X_MIN,Y_MIN,X_MAX,Y_MAX)
    # the model predicts the bounding boxes, its offsets, its errors in the shape of (Xc,Yc,W,H)

    # Split the anchor box into the individual coordinates
    x_c,y_c,w,h = tf.split(anchor_boxes, num_or_size_splits = 4, axis= -1)

    # Calculating the coordinates in XY format
    x_min = x_c - (w/2)
    x_max = x_c + w/2
    y_min = y_c - h/2
    y_max = y_c + h/2

    converted_boxes = tf.concat([x_min,y_min,x_max,y_max],axis=-1) # Stacking the boxes together into one vector
    
    return converted_boxes

# Calculating the bounding box deltas for the predicted boxes, anchor boxes, ground truth boxes
def calculate_bounding_box_deltas_between_pred_and_anchor_box(anchor_boxes,pred_boxes):
    """
    Calculate the bounding box deltas 't' between the predicted anchor box offsets and the anchor boxes generated

    Parameters:
    ---------
    anchor_boxes : Tensor
        Tensor Of Generated Anchor Boxes (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_OF_ANCHORS,4)

    pred_boxes : Tensor
        Tensor Of Predicted Anchor Box Offsets By RPN (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_OF_ANCHORS,4)
    Returns:
    -------
    t: Tensor
        Bounding Box offsets between the predicted RPN anchor boxes and the predefined anchor boxes (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_OF_ANCHORS,4)       
    """
    # Splitting the anchor boxes into xc,yc,w,h separately
    anc_xc,anc_yc,anc_w,anc_h = tf.split(anchor_boxes,num_or_size_splits = 4,axis=-1)

    # Splitting the predicted boxes into xc,yc,w,h separately
    pred_xc,pred_yc,pred_w,pred_h = tf.split(pred_boxes,num_or_size_splits = 4,axis=-1)

    # According to the paper the deltas, the coefficients are calculated for the anchor box as well as
    # ground truth box called t & t*. These are only for positive anchors only
    tx = (pred_xc - anc_xc) / (anc_w + 1e-6)
    ty = (pred_yc - anc_yc) / (anc_h + 1e-6)
    tw = tf.math.log(tf.maximum((pred_w/(anc_w + 1e-6)),1e-6)) # Adding an epsilon value to prevent a NaN being made from division by zero
    th = tf.math.log(tf.maximum((pred_h/(anc_h + 1e-6)),1e-6))

    # Squeezing the last dimension
    tx = tf.squeeze(tx,axis=-1)
    ty = tf.squeeze(ty,axis=-1)
    tw = tf.squeeze(tw,axis=-1)
    th = tf.squeeze(th,axis=-1)
    
    t = tf.stack([tx,ty,tw,th],axis=-1)
    
    return t

# Calculating the bounding box deltas for the anchor boxes, ground truth boxes
def calculate_bounding_box_deltas_between_gt_boxes(gt_boxes,anchor_boxes,iou_matrix,object_labels):
    """
    Calculate the bounding box deltas 't*' between the ground truth boxes and their corresponding anchor boxes
    
    Parameters:
    ---------
    gt_boxes : Tensor
        Tensor Of Generated Anchor Boxes (B,GT_BOXES,4)

    anchor_boxes : Tensor
        Tensor Of Predicted Anchor Box Offsets By RPN (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_OF_ANCHORS,4) in the form of (Xc,Yc,W,H)

    iou_matrix : Tensor
        Tensor of IOU Matrix for the anchor boxes (B,GT_BOXES,NUM_TOTAL_ANCHORS)

    object_labels : Tensor
        Tensor Of Object Labels (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_ANCHORS,1)
        
    Returns:
    -------
    t: Tensor
        Bounding Box offsets between the ground truth boxes and the predefined anchor boxes (NUM_POS_ANCHORS,4)
    """

    batch_size , feature_map_width, feature_map_height,number_of_anchors_per_pixel , _ = anchor_boxes.shape
    object_labels = tf.reshape(tf.squeeze(object_labels,axis=-1),[object_labels.shape[0],object_labels.shape[1]*object_labels.shape[2]*object_labels.shape[3],1])
    anchor_boxes = tf.reshape(anchor_boxes,[anchor_boxes.shape[0],anchor_boxes.shape[1]*anchor_boxes.shape[2]*anchor_boxes.shape[3],4])

    # We need to gather the bounding boxes
    positive_mask = tf.squeeze(object_labels,axis=-1) == 1
    positive_indices = tf.where(positive_mask)
    positive_anchor_boxes = tf.gather_nd(anchor_boxes,positive_indices)

    # Now we need to gather the ground truth box which these positive anchors are associated with
    gt_box_per_anchor_box_indices = tf.argmax(iou_matrix,axis=1)
    positive_gt_indices = tf.boolean_mask(gt_box_per_anchor_box_indices,positive_mask)
    batch_indices = positive_indices[:, 0]
    batch_and_gt_indices = tf.stack([batch_indices, positive_gt_indices], axis=-1)
    corresponding_gt_boxes = tf.gather_nd(gt_boxes,batch_and_gt_indices)

    corresponding_gt_boxes = convert_xy_boxes_to_center_format(corresponding_gt_boxes) # Convert the coordinate system from (X_MIN,Y_MIN,X_MAX,Y_MAX) to (Xc,Yc,W,H)
    positive_anchor_boxes = convert_xy_boxes_to_center_format(positive_anchor_boxes)   # Convert the coordinate system from (X_MIN,Y_MIN,X_MAX,Y_MAX) to (Xc,Yc,W,H)

    # Splitting the anchor boxes into xc,yc,w,h separately
    gt_xc,gt_yc,gt_w,gt_h = tf.split(corresponding_gt_boxes,num_or_size_splits = 4,axis=-1)

    # Splitting the predicted boxes into xc,yc,w,h separately
    pred_xc,pred_yc,pred_w,pred_h = tf.split(positive_anchor_boxes,num_or_size_splits = 4,axis=-1)

    # According to the paper the deltas, the coefficients are calculated for the anchor box as well as
    # ground truth box called t & t*. These are only for positive anchors only
    tx = (gt_xc - pred_xc) / (pred_w + 1e-6)
    ty = (gt_yc - pred_yc) / (pred_h + 1e-6)
    tw = tf.math.log(tf.maximum(((gt_w + 1e-6)/pred_w),1e-6))
    th = tf.math.log(tf.maximum(((gt_h + 1e-6)/pred_h),1e-6))

    # Squeezing the last dimension
    tx = tf.squeeze(tx,axis=-1)
    ty = tf.squeeze(ty,axis=-1)
    tw = tf.squeeze(tw,axis=-1)
    th = tf.squeeze(th,axis=-1)
    
    t = tf.stack([tx,ty,tw,th],axis=-1)

    return t

# Calculating the smooth L1 loss that happens between the offsets t and t*
def smooth_l1_loss(prediction,ground_truth,beta = 1.0):
    """
    Calculates the Smooth L1 loss based on the formula
    
    Parameters:
    ---------
    prediction: Tensor
        Tensor of the predicted values

    ground_truth: Tensor
        Tensor of the ground truth values

    beta: Float
        Value for the loss thresholding to apply the appropriate loss calculation on the value

    Returns:
    -------
    loss: Float
        L1 loss value for the tensors
    """
    abs_loss = tf.math.abs(prediction - ground_truth)
    # Loss is calculated depending on if the absolute loss is less than beta value
    loss = tf.reduce_mean(tf.where(abs_loss < beta,(0.5*(abs_loss ** 2))/beta,(beta * (abs_loss - 0.5*beta)))) # Mean since the formula uses MAE formula
    
    return loss

# Calculating the bounding box regression loss (L_reg) for the bounding box offsets of the model and the anchor boxes
def calculate_bounding_box_regression_loss(anchor_boxes,offsets,gt_boxes,iou_matrix,object_labels,anchor_scaling_stride = 16):
    """
    Calculates the bounding box regression loss for the offsets
    
    Parameters:
    ---------
    anchor_boxes: Tensor
        Tensor of the anchor boxes (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_ANCHORS,4)

    offsets: Tensor
        Tensor of the predicted offsets (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_ANCHORS,4)

    gt_boxes: Tensor
        Tensor of the ground truth boxes (B,NUM_GT_BOXES,4)

    iou_matrix: Tensor
        Tensor of the IoU values for the anchor boxes (B,GT_BOXES,NUM_TOTAL_ANCHORS)

    object_labels: Tensor
        Tensor of the object classes (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_OF_ANCHORS_PER_PIXEL,1)

    anchor_scaling_stride: Float
        Value for scaling the anchor boxes from the feature map space to the image space

    Returns:
    -------
    loss: Float
        L1 loss value for the offsets t and t*
    """
    scaled_anchor_boxes = anchor_boxes * anchor_scaling_stride

    # # After conversion the offset t* is calculated
    t_star_offset = calculate_bounding_box_deltas_between_gt_boxes(gt_boxes,scaled_anchor_boxes,iou_matrix,object_labels)

    # After calculating t and t* the positive anchors from t are retained and smooth l1 loss is calculated amongst them
    positive_mask = tf.squeeze(object_labels,axis=-1) == 1 # Squeezed the last dimension to make it more intuitive, created a binary mask for the positive anchors
    positive_offsets = tf.boolean_mask(offsets,positive_mask) # Gathered all the positive anchors in the data

    # # After gathering the offsets for the positive anchor boxes, the smooth l1 loss can be calulcated for it
    loss = smooth_l1_loss(positive_offsets,t_star_offset)

    return loss

# Calculate the loss of the RPN model using the objectness score and the bounding box regression score
def calculate_rpn_loss(anchor_boxes,offsets,gt_boxes,iou_matrix,object_labels,objectness_scores,anchor_scaling_stride = 16,batch_sample_per_image = 256,positive_composition_ratio = 0.5,lambda_ = 10):
    """
    Calculates the total loss of the RPN which includes both classification and regression losses
    
    Parameters:
    ---------
    anchor_boxes: Tensor
        Tensor of the anchor boxes (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_ANCHORS,4)

    offsets: Tensor
        Tensor of the predicted offsets (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_ANCHORS,4)

    gt_boxes: Tensor
        Tensor of the ground truth boxes (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_ANCHORS,4)

    iou_matrix: Tensor
        Tensor of the IoU values for the anchor boxes (B,GT_BOXES,NUM_TOTAL_ANCHORS)

    object_labels: Tensor
        Tensor of the object classes (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_OF_ANCHORS_PER_PIXEL,1)

    anchor_scaling_stride: Float
        Value for scaling the anchor boxes from the feature map space to the image space

    batch_sample_per_image : Int
        Sample Size for each image that is used to limit the sampling size that can be changed if the errors are too large
        
    positive_composition_ratio: Float
        Positive composition ratio, default value is 50% but it works with up to the limit if the positives are low, the values needs to be changed

    lambda_: Float
        Balancing parameter based on the research paper to not allow one loss to dominate the other

    Returns:
    -------
    total_loss: Float
        Total loss value for the RPN model
    """ 
    # Calculate the number of positive_anchors
    # num_positive_anchor_boxes = tf.reduce_sum(tf.cast(object_labels == 1, tf.float32))
    # num_negative_anchor_boxes = tf.reduce_sum(tf.cast(object_labels == 0, tf.float32))

    # Calculate the normalizing terms
    # n_cls = tf.maximum(num_positive_anchor_boxes + num_negative_anchor_boxes, 1e-6)
    # n_reg = tf.maximum(num_positive_anchor_boxes, 1e-6)
    object_labels = rpn_sample_anchors(object_labels,batch_sample_per_image,positive_composition_ratio)
    objectness_loss = calculate_objectness_loss(objectness_scores[...,1],object_labels)
    regression_loss = calculate_bounding_box_regression_loss(anchor_boxes,offsets,gt_boxes,iou_matrix,object_labels,anchor_scaling_stride = anchor_scaling_stride)
    # Calculate the total loss based on the paper
    total_loss =  (objectness_loss) + (lambda_ * regression_loss) 

    return total_loss

def apply_bounding_box_deltas(anchor_boxes,deltas,max_width = 800, max_height = 800):
    # Split the proposal coordinates (xc,yc,w,h)
    x_c,y_c,w,h = tf.split(anchor_boxes, num_or_size_splits = 4,axis = -1) # Splitting the RoI Coordinates

    t_x,t_y,t_w,t_h = tf.split(deltas, num_or_size_splits = 4,axis = -1) # Splitting the Offsets
    
    # Clipping the t_w and t_h values to prevent too large values that can cause overflow in exp function
    t_w = tf.clip_by_value(t_w,-4,4)
    t_h = tf.clip_by_value(t_h,-4,4)

    x_p = (t_x * w) + x_c
    y_p = (t_y * h) + y_c
    w_p = tf.math.exp(t_w) * w
    h_p = tf.math.exp(t_h) * h

    x1 = tf.clip_by_value(x_p - 0.5 * w_p, 0.0, max_width)
    y1 = tf.clip_by_value(y_p - 0.5 * h_p, 0.0, max_height)
    x2 = tf.clip_by_value(x_p + 0.5 * w_p, 0.0, max_width)
    y2 = tf.clip_by_value(y_p +
                          0.5 * h_p, 0.0, max_height)

    proposals = tf.concat([x1,y1,x2,y2],axis=-1)
    
    return proposals

# Function to refine the region of interests based on the offsets predicted by the RPN
def refine_region_of_interests(anchor_boxes,offsets,feature_map_width = 50, feature_map_height = 50):
    """
    Calculates the Region of Interests (RoI) for the positive anchor boxes so that they can be then used in the RoI pooling.
    
    Parameters:
    ---------
    anchor_boxes: Tensor
        Tensor of the positive anchor boxes in the batch in the shape of (NUM_POSITIVE_ANCHORS_BOXES,4)

    offsets: Tensor
        Tensor of the offsets for the positive anchor boxes in the shape of (NUM_POSITIVE_ANCHOR_BOXES,4)

    feature_map_width : Int
        The width of the feature map coordinate space (default= 50)

    feature_map_height : Int
        The height of the feature map coordinate space (default = 50)
        
    Returns:
    -------
    roi_proposals: Tesnor
        ROI Proposals that are created by adding the anchor boxes and their refinements in the shape (NUM_POSITIVE_ANCHOR_BOXES,4)
    """ 
    # Take the anchor boxes and add the offsets to them to calculate the actual ROI in feature map space
    x_c,y_c,w,h = tf.split(anchor_boxes,num_or_size_splits=4,axis=-1)
    t_xc,t_yc,t_w,t_h = tf.split(offsets,num_or_size_splits=4,axis=-1)

    t_w = tf.clip_by_value(t_w,-4,4)
    t_h = tf.clip_by_value(t_h,-4,4)

    # Adding them anchor boxes with the offsets in the opposite way while calulating the differences in the offsets according to the paper.
    roi_xc = x_c + (t_xc * w)
    roi_yc = y_c + (t_yc * h)
    roi_w = tf.exp(t_w) * w
    roi_h = tf.exp(t_h) * h

    roi_proposals = tf.concat([roi_xc,roi_yc,roi_w,roi_h],axis=-1) # Concat them to create them to box coordinates

    return roi_proposals     # Created the RoI's, they need to be clipped when converting to the xy-coordinate system

# Function to convert the bounding box coordinates from (xc,yc,w,h) into (x_min,y_min,x_max,y_max) for the 
# tensorflow.image.crop_and_resize function which expects that format
def convert_rois_to_x_y_coordinates(rois, feature_map_width = 50, feature_map_height = 50):
    """
    Converting the bounding boxes from (xc,yc,w,h) to (x_min,y_min,x_max,y_max) for use in the tensorflow method of tf.image.crop_and_resize
    Note: Output bounding boxes are in feature map space
    Parameters:
    ---------
    rois: Tensor
        Tensor of the positive anchor boxes in the batch in the shape of (NUM_POSITIVE_ANCHORS_BOXES,4)

    feature_map_width : Int
        The width of the feature map coordinate space (default= 50)

    feature_map_height : Int
        The height of the feature map coordinate space (default = 50)
        
    Returns:
    -------
    bounding_boxes: Tesnor
        Bounding boxes in the format of (x_min,y_min,x_max,y_max)
    """ 
    # Need to convert the bounding boxes from format (xc,yc,w,h) to (x_min,y_min,x_max,y_max)
    x_c,y_c,w,h = tf.split(rois,num_or_size_splits = 4,axis=-1)

    # The x-axis is not changed since the axis orientation in image space does not change
    # The y-axis is changed since the origin point is in top left and not bottom right
    
    x_min = tf.maximum((x_c - w/2) , 0) # Top left x-value
    y_min = tf.maximum((y_c - h/2) , 0) # Top left y-value
    x_max = tf.minimum((x_c + w/2) , feature_map_width) # Bottom right x-value
    y_max = tf.minimum((y_c + h/2) , feature_map_height) # Bottom right y-value

    bounding_boxes = tf.concat([x_min,y_min,x_max,y_max],axis=-1) # Converting the format (x_min,y_min,x_max,y_max)
    return bounding_boxes

# Function to divide the Feature map into grids and blocks that can be pooled to be used in the RoI Head
def divide_roi_into_grids(feature_map,bounding_boxes,box_indices,grid_row_size,grid_col_size,image_width = 800, image_height = 800):
    """
    Dividing the RoI's into grids and blocks that are pooled to be used in the RoI Head for the loss
    Parameters:
    ---------
    feature_map: Tensor
        Feature Map from the RPN (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,CHANNELS)

    bounding_boxes : Tensor
        Tensor of bounding boxes coordinates in the format (X_MIN,Y_MIN,X_MAX,Y_MAX) and the shape of (NUM_RoIs,4)

    box_indices : [Int]
        Indices for the bounding boxes

    grid_row_size: Int
        Row size for the cropped and resized grid

    grid_col_size: int
        Column size for the cropped and resized grid
        
    Returns:
    -------
    roi: Tesnor
        RoI's in the cropped and resized shape of (NUM_ROI,GRID_ROW_SIZE,GRID_COL_SIZE,CHANNELS)
    """

    # Splitting the bounding boxes
    x_min,y_min,x_max,y_max = tf.split(bounding_boxes, num_or_size_splits = 4, axis=-1)

    # print(f'X_MIN: {x_min}, Y_MIN: {y_min}, X_MAX: {x_max}, Y_MAX: {y_max}')

    # Tensorflow crop and resize requires the 'boxes' to be normalized between [0,1]
    x_min_normalized = x_min / tf.cast(image_width,tf.float32)
    y_min_normalized = y_min / tf.cast(image_height,tf.float32)
    x_max_normalized = x_max / tf.cast(image_width,tf.float32)
    y_max_normalized = y_max / tf.cast(image_height,tf.float32)

    # print(f'X_MIN_NORM: {x_min_normalized}, Y_MIN_NORM: {y_min_normalized}, X_MAX_NORM: {x_max_normalized}, Y_MAX_NORM: {y_max_normalized}')

    # # Stacking the normalized tensor to be used in the image.crop_and_resize
    normalized_bounding_boxes = tf.concat([y_min_normalized, x_min_normalized, y_max_normalized, x_max_normalized],axis=-1)

    # Removing the boxes with zero height and width since they are the padded boxes and edge leaning boxes are not chucked too
    y1,x1,y2,x2 = tf.unstack(normalized_bounding_boxes,axis=-1)
    
    valid_area = tf.logical_and((x2 - x1) > 0.0, (y2 - y1) > 0.0) # Isolating the boxes where they are padded
    
    padded_boxes = tf.reduce_all(tf.equal(normalized_bounding_boxes,0.0), axis=-1)
    
    valid_mask = tf.logical_and(valid_area,tf.logical_not(padded_boxes))

    normalized_bounding_boxes = tf.boolean_mask(normalized_bounding_boxes,valid_mask) # Gathering and Stacking them together

    box_indices =  tf.boolean_mask(box_indices,valid_mask) # Filtering the box indices too in case there are padded boxes

    # Need to make sure that the function receives the coordinates in the shape (Y_MIN,X_MIN,Y_MAX,X_MAX)
    roi = tf.image.crop_and_resize(feature_map,normalized_bounding_boxes,tf.cast(box_indices,tf.int32),(grid_row_size,grid_col_size))

    return roi,box_indices,valid_mask

def divide_proposals_into_rois(feature_map,bounding_boxes,box_indices,grid_row_size,grid_col_size,image_width = 800, image_height = 800):
    """
    Dividing the RoI's into grids and blocks that are pooled to be used in the RoI Head for the loss
    Parameters:
    ---------
    feature_map: Tensor
        Feature Map from the RPN (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,CHANNELS)

    bounding_boxes : Tensor
        Tensor of bounding boxes coordinates in the format (X_MIN,Y_MIN,X_MAX,Y_MAX) and the shape of (NUM_RoIs,4)

    grid_row_size: Int
        Row size for the cropped and resized grid

    grid_col_size: int
        Column size for the cropped and resized grid
        
    Returns:
    -------
    roi: Tensor
        RoI's in the cropped and resized shape of (NUM_ROI,GRID_ROW_SIZE,GRID_COL_SIZE,CHANNELS)
    """

    # Splitting the bounding boxes
    x_min,y_min,x_max,y_max = tf.split(bounding_boxes, num_or_size_splits = 4, axis=-1)

    # print(f'X_MIN: {x_min}, Y_MIN: {y_min}, X_MAX: {x_max}, Y_MAX: {y_max}')

    # Tensorflow crop and resize requires the 'boxes' to be normalized between [0,1]
    x_min_normalized = x_min / tf.cast(image_width,tf.float32)
    y_min_normalized = y_min / tf.cast(image_height,tf.float32)
    x_max_normalized = x_max / tf.cast(image_width,tf.float32)
    y_max_normalized = y_max / tf.cast(image_height,tf.float32)

    # print(f'X_MIN_NORM: {x_min_normalized}, Y_MIN_NORM: {y_min_normalized}, X_MAX_NORM: {x_max_normalized}, Y_MAX_NORM: {y_max_normalized}')

    # # Stacking the normalized tensor to be used in the image.crop_and_resize
    normalized_bounding_boxes = tf.concat([y_min_normalized, x_min_normalized, y_max_normalized, x_max_normalized],axis=-1)

    # Removing the boxes with zero height and width since they are the padded boxes and edge leaning boxes are not chucked too
    y1,x1,y2,x2 = tf.unstack(normalized_bounding_boxes,axis=-1)
    
    valid_area = tf.logical_and((x2 - x1) > 0.0, (y2 - y1) > 0.0) # Isolating the boxes where they are padded
    
    padded_boxes = tf.reduce_all(tf.equal(normalized_bounding_boxes,0.0), axis=-1)
    
    valid_mask = tf.logical_and(valid_area,tf.logical_not(padded_boxes))

    normalized_bounding_boxes = tf.boolean_mask(normalized_bounding_boxes,valid_mask) # Gathering and Stacking them together
    
    box_indices =  tf.boolean_mask(box_indices,tf.reshape(valid_mask,[-1])) # Stacking and flattening the mask for easier box indices

    # Need to make sure that the function receives the coordinates in the shape (Y_MIN,X_MIN,Y_MAX,X_MAX)
    roi = tf.image.crop_and_resize(feature_map,normalized_bounding_boxes,tf.cast(box_indices,tf.int32),(grid_row_size,grid_col_size))

    return roi, box_indices, valid_mask

# Function for RoI pooling, that will take the positive anchor boxes and making it into a fixed size to be flattened
# and used in the RoI head
def roi_pooling(feature_map,positive_anchor_boxes,positive_indices,offsets_for_anchor_boxes,output_size_x = 7,output_size_y = 7,stride = 16):
    """
    RoI pooling for the positive anchor boxes/regions generated by the RPN
    Parameters:
    ---------
    feature_map: Tensor
        Feature Map from the RPN (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,CHANNELS)

    positive_anchor_boxes : Tensor
        Tensor of bounding boxes coordinates in the format (X_MIN,Y_MIN,X_MAX,Y_MAX) and the shape of (B,NUM_POS_ANCHOR_BOXES,4)

    positive_indices : Tensor
        Indices of the positive anchor boxes in the shape of (NUM_POS_ANCHORS,4) in the format of (B,ROW_NUM,COL_NUM,NUM_OF_ANCHORS_PER_PIXEL)

    offsets_for_anchor_boxes: Tensor
        Tensor of bounding boxes coordinates in the format (X_MIN,Y_MIN,X_MAX,Y_MAX) and the shape of (NUM_POS_ANCHOR_BOXES,4)

    output_size_x: int
        Output row size for the RoI's

    output_size_x: int
        Output column size for the RoI's
        
    Returns:
    -------
    roi: Tensor
        RoI's in the cropped and resized shape of (B,NUM_ROI,GRID_ROW_SIZE,GRID_COL_SIZE,CHANNELS)

    bounding_boxes: Tensor
        Bounding boxes for the the RoI's in XY Coordinate system in the shape of (B,NUM_POS_ANCHOR_BOXES,4)
    """

    # Convert the bounding boxes from (X_MIN,Y_MIN,X_MAX,Y_MAX) to (Xc,Yc,W,H) -> Feature Map Space to Image Space
    converted_positive_anchor_boxes = convert_xy_boxes_to_center_format(positive_anchor_boxes * stride)
    
    # Need to create the RoI's before pooling them in the correct size
    region_of_interests = refine_region_of_interests(converted_positive_anchor_boxes,offsets_for_anchor_boxes)
    bounding_boxes = convert_center_format_boxes_to_xy_coordinate(region_of_interests) # Needed since offsets can be only done in the center format
   

    # # Divide RoI's into Grids
    blocks,positive_indices,valid_mask = divide_roi_into_grids(feature_map,bounding_boxes,positive_indices[:,:,0],output_size_x,output_size_y)

    # Need to stack them back again into batched padded tensors for image wise
    image_rois_split = []
    for image in range(feature_map.shape[0]):
        # Creating a mask for the particular image
        image_mask = positive_indices == image
        # Getting the RoI's that align with this mask
        image_rois = tf.boolean_mask(blocks,image_mask)

        image_rois_split.append(image_rois)

    # Padding the tensors so that they can be stacked
    roi_lengths = tf.stack([tf.shape(dim)[0] for dim in image_rois_split])
    num_of_max_rois = tf.reduce_max(roi_lengths)
    padded_tensors = [tf.pad(tensor, paddings = [[0,num_of_max_rois - tf.shape(tensor)[0]],[0,0],[0,0],[0,0]]) for tensor in image_rois_split]

    # Stacking the two tensors together
    stacked_roi_tensors = tf.stack(padded_tensors,axis=0)

    # Filtering the bounding boxes to remove the degenerate boxes that may exist
    
    valid_boxes = []
    for image in range(bounding_boxes.shape[0]):
        # Get the indices that are for that image
        image_mask = valid_mask[image]
        valid_boxes.append(tf.boolean_mask(bounding_boxes[image],image_mask))
    
    box_lengths = tf.stack([tf.shape(dim)[0] for dim in valid_boxes])
    tf.debugging.assert_equal(box_lengths, roi_lengths)
    
    padded_tensors = [tf.pad(tensor, paddings = [[0,num_of_max_rois - tf.shape(tensor)[0]],[0,0]]) for tensor in valid_boxes]

    bounding_boxes = tf.stack(padded_tensors,axis=0)
    
    return stacked_roi_tensors, bounding_boxes

# Function for RoI pooling, that will take the positive anchor boxes and making it into a fixed size to be flattened
# and used in the RoI head
def roi_align(feature_map,sampled_proposals,output_size_x = 7,output_size_y = 7,stride = 16):
    """
    RoI Align for the positive anchor boxes/regions generated by the RPN
    
    Parameters:
    ---------
    feature_map: Tensor
        Feature Map from the RPN (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,CHANNELS)

    sampled_proposals : Tensor
        Tensor of sampled proposals from the RPN using the (B,NUM_SAMPLES,4)

    output_size_x: int
        Output row size for the RoI's

    output_size_x: int
        Output column size for the RoI's
        
    Returns:
    -------
    roi: Tensor
        RoI's in the cropped and resized shape of (B,NUM_ROI,GRID_ROW_SIZE,GRID_COL_SIZE,CHANNELS)
    """

    proposal_indices = tf.repeat(tf.range(sampled_proposals.shape[0]),sampled_proposals.shape[1])

    # Divide into the grids
    rois, box_indices, valid_mask = divide_proposals_into_rois(feature_map,sampled_proposals,proposal_indices,output_size_x,output_size_y)

    # Creating the RoIs back into the images
    # rois = tf.reshape(rois,[sampled_proposals.shape[0],sampled_proposals.shape[1],output_size_x,output_size_y,-1])
    
    # Need to stack them back again into batched padded tensors for image wise
    image_rois_split = []
    image_proposals_split = []
    for image in range(feature_map.shape[0]):
        # Creating a mask for the particular image
        image_mask = box_indices == image
        # Getting the RoI's that align with this mask
        image_rois = tf.boolean_mask(rois,image_mask)
        image_proposals = tf.boolean_mask(sampled_proposals[image],valid_mask[image])
        
        image_rois_split.append(image_rois)
        image_proposals_split.append(image_proposals)

    # Padding the tensors so that they can be stacked
    roi_lengths = tf.stack([tf.shape(dim)[0] for dim in image_rois_split])
    num_of_max_rois = tf.reduce_max(roi_lengths)
    
    padded_tensors = [tf.pad(tensor, paddings = [[0,num_of_max_rois - tf.shape(tensor)[0]],[0,0],[0,0],[0,0]]) for tensor in image_rois_split]

    # Stacking the two tensors together
    stacked_roi_tensors = tf.stack(padded_tensors,axis=0)
    
    box_lengths = tf.stack([tf.shape(dim)[0] for dim in image_proposals_split])
    tf.debugging.assert_equal(box_lengths, roi_lengths)
    
    padded_tensors = [tf.pad(tensor, paddings = [[0,num_of_max_rois - tf.shape(tensor)[0]],[0,0]]) for tensor in image_proposals_split]

    stacked_roi_coordinates = tf.stack(padded_tensors,axis=0)
    
    return stacked_roi_tensors, stacked_roi_coordinates

# Function to assign the RoI's to the ground truth boxes based on the IoU scores
def assign_roi_to_ground_truth_box(ground_truth_boxes,roi_coordinates,ground_truth_labels,fg_thresh = 0.5,bg_thresh = 0.3,image_stride = 16):
    """
    Assign RoI to Ground truth boxes
    Parameters:
    ---------
    ground_truth_boxes: Tensor
        Ground truth box tensor (B,NUM_OF_GT_BOXES,4)

    roi_coordinates : Tensor
        Tensor of RoI coordinates in the format (X_MIN,Y_MIN,X_MAX,Y_MAX) and the shape of (B,NUM_OF_ROIS,4)

    ground_truth_labels : Tensor
        Tensor of ground truth labels in the shape of (B,NUM_OF_GT_BOXES)

    fg_thresh: Float
        Foreground threshold when calculating the labels for the foreground objects
    
    bg_thresh: Float
       Background threshold when calculating the labels for the background objects

    image_stride: Int
        Stride of the model to convert from Feature Map Shape to Image Space, default value of 16
        
    Returns:
    -------
    roi_labels: Tensor
        Tensor of labels for each RoI in the shape of (B,NUM_OF_ROI)

    best_gt_box_for_each_anchor_box: Tensor
        Tensor of best GT box for each RoI based on IoU in the shape of (B,NUM_OF_ROI)

    max_iou_per_anchor_box: Tensor
        Tensor of max IoU per anchor box based on the IoU score in the shape of (B,NUM_OF_ROI)
    """
    x1, y1, x2, y2 = tf.split(roi_coordinates, 4, axis=-1)
    valid_roi_mask = tf.squeeze((x2 > x1) & (y2 > y1), axis=-1)
    
    # Calculate the IoU score
    iou_score = IOU_scores(ground_truth_boxes,roi_coordinates)

    # Get the best anchor box for each ground truth box
    best_roi_for_each_gt_boxes = tf.argmax(iou_score,axis = -1)

    # Getting the best ground truth for anchor boxes
    best_gt_box_for_each_anchor_box = tf.argmax(iou_score,axis = 1)

    # Getting Max IOU for Each RoI
    max_iou_per_anchor_box = tf.reduce_max(iou_score, axis=1)

    # Check if IOU value is more than the threshold
    roi_labels = tf.gather(ground_truth_labels,best_gt_box_for_each_anchor_box,batch_dims=1)

    ignore = tf.cast(-1, dtype=roi_labels.dtype)
    background = tf.cast(0,  dtype=roi_labels.dtype)

    pos_mask = max_iou_per_anchor_box >= fg_thresh
    bg_mask  = max_iou_per_anchor_box <= bg_thresh
    

    roi_labels = tf.where(pos_mask, roi_labels, ignore)
    roi_labels = tf.where(bg_mask,  background,         roi_labels)

    roi_labels = tf.where(valid_roi_mask, roi_labels, ignore)

    if tf.reduce_sum(tf.cast(roi_labels > 0, tf.int32)) == 0:
        tf.print("⚠️ Warning: All RoIs are background for this sample.")

    num_fg = tf.reduce_sum(tf.cast(roi_labels > 0, tf.int32))
    tf.print("Foreground RoIs in batch:", num_fg)
    
    return roi_labels, best_gt_box_for_each_anchor_box, max_iou_per_anchor_box

# Function to match the ground truth boxes to the RoI coordinates
def match_gt_box_to_roi_coordinate(ground_truth_boxes,best_gt_box_for_each_anchor_box):
    """
    Match ground truth boxes to the RoI coordinates
    Parameters:
    ---------
    ground_truth_boxes: Tensor
        Ground truth box tensor (B,NUM_OF_GT_BOXES,4)

    best_gt_box_for_each_anchor_box: Tensor
        Tensor of best GT box for each RoI based on IoU in the shape of (B,NUM_OF_ROI)
    
    Returns:
    -------
    matched_gt_boxes : Tensor
        Tensor of matched GT boxes for the RoI Coordinates in the shape of (B, MAX_NUM_OF_ROI,4)
    """
    return tf.gather(ground_truth_boxes,best_gt_box_for_each_anchor_box,batch_dims=1)

def calculate_bounding_box_deltas_between_roi_and_ground_truth_box(ground_truth_boxes,roi_coordinates, roi_labels,image_space_stride = 16):
    """
    Calculate bounding box deltas between RoI coordinate boxes and ground truth boxes
    Parameters:
    ---------
    ground_truth_boxes: Tensor
        Ground truth box tensor (B,NUM_OF_GT_BOXES,4)

    roi_coordinates : Tensor
        Tensor of RoI coordinates in the format (X_MIN,Y_MIN,X_MAX,Y_MAX) and the shape of (B,NUM_OF_ROIS,4) in image space
        
    Returns:
    -------
    filtered_t_offsets: Tensor
        Tensor of labels for each RoI in the shape of (B,NUM_OF_GT_BOXES,4)
    """
    # Converting the RoI coordinate from xy-format to the center format
    center_roi_coordinates = convert_xy_boxes_to_center_format(roi_coordinates)
    # Converting the ground truth boxes from xy-format to center format
    center_ground_truth_boxes = convert_xy_boxes_to_center_format(ground_truth_boxes)

    offsets_split = []

    for batch in range(tf.shape(center_roi_coordinates)[0]):
        # Calculate the offsets using the formula from the paper
        gt_center_x, gt_center_y, gt_w, gt_h = tf.split(center_ground_truth_boxes[batch], num_or_size_splits = 4, axis = -1) # Splitting ground truth boxes
        roi_center_x, roi_center_y, roi_w, roi_h = tf.split(center_roi_coordinates[batch], num_or_size_splits = 4, axis = -1) # Splitting the RoI's

        # Calculating the offsets
        t_x = (gt_center_x - roi_center_x)/(roi_w + 1e-6)
        t_y = (gt_center_y - roi_center_y)/(roi_h + 1e-6)
        t_w = tf.math.log(gt_w/(roi_w + 1e-6))
        t_h = tf.math.log(gt_h/(roi_h + 1e-6))

        t = tf.concat([t_x,t_y,t_w,t_h],axis=-1)

        # Calculating a valid mask based on the rois
        x1,y1,x2,y2 = tf.unstack(roi_coordinates[batch],axis=-1)
    
        valid_area = tf.logical_and((x2 - x1) > 0.0, (y2 - y1) > 0.0) # Isolating the boxes where they are padded
    
        padded_boxes = tf.reduce_all(tf.equal(roi_coordinates[batch],0.0), axis=-1)
    
        valid_mask = tf.logical_and(valid_area,tf.logical_not(padded_boxes))
        
        filtered_t_offsets = tf.boolean_mask(t,valid_mask)

        offsets_split.append(filtered_t_offsets)

    num_of_max_offsets = tf.reduce_max([tf.shape(dim)[0] for dim in offsets_split])
    padded_tensors = [tf.pad(tensor, paddings = [[0,num_of_max_offsets - tf.shape(tensor)[0]],[0,0]]) for tensor in offsets_split]

    stacked_offsets = tf.stack(padded_tensors,axis=0)
        
    return stacked_offsets, roi_labels

# Turn image wise information into batchwise
# Mask out -1 RoI's and classification scores
# Stack them into one batch. Calculate the SparseCategoricalEntropyLoss on these new batches
def calculate_roi_head_classification_loss(roi_labels,classification_head):
    """
    Calculating the RoI Head Classification Loss

    Parameters:
    ---------
    roi_labels : Tensor
        Tensor of RoI Labels in the shape of (B,NUM_ROIS)

    classification_head: Tensor
        Tensor of classification scores from the RoI Head (B,NUM_ROIS,NUM_CLASSES)

    Returns:
    -------
    cls_loss: Float
        RoI classification loss
    """

    # Removing the -1 RoI's
    valid_mask = roi_labels != -1
    valid_roi_labels = tf.boolean_mask(roi_labels,valid_mask)

    num_valid = tf.reduce_sum(tf.cast(valid_mask,tf.float32))
    if tf.equal(num_valid,0.0):
        return tf.constant(0.0, dtype=tf.float32)

    # Using the same mask to eliminate the Classification Scores from RoI
    valid_classification_head = tf.boolean_mask(classification_head,valid_mask)

    # Calculate the loss between the two
    loss_fn = SparseCategoricalCrossentropy(from_logits = True,reduction=tf.keras.losses.Reduction.NONE)
    cls_loss = tf.reduce_sum(loss_fn(valid_roi_labels,valid_classification_head))/(num_valid + 1e-8)
    
    return cls_loss

# Calculate the Smooth L1 Loss between the t* and the predicted values from the RoI Head Regression head

def calculate_roi_head_regression_loss(t_star,regression_head,roi_labels, number_of_classes):
    """
    Calculates the regression loss between the t_star calculated and the predicted values from RoI Head Regression head.
    
    Parameters:
    ---------
    t_star : Tensor
        Tensor of filtered calculated offsets (t_star) for the foreground RoI's in the shape of (B,NUM_FOREGROUND_ROIs,4)

    regression_head: Tensor
        Tensor of predicted regression offsets by the RoI Head (B,NUM_ROIs,NUM_OF_CLASSES*4)

    roi_labels : Tensor
        Tensor of filtered RoI Labels for the foreground RoI's in the shape of (NUM_FOREGROUND_ROIs,)

    number_of_classes: Int
        Number of classes in the RoI Head

    Returns:
    -------
    deltas_loss: Float
        RoI bounding box delta loss value
    """
    # # Reshape the regression head output
    # flattened_regression_head = tf.reshape(regression_head, [-1, number_of_classes, 4])

    # # If there is no foreground object
    # if tf.shape(t_star)[0] == 0:
    #     return tf.constant(0.0, dtype=tf.float32)

    # # Calculating the coordinate indices to select the right class for each RoI
    # index = tf.range(t_star.shape[0],dtype=tf.int32)

    # # Stack the indices together for selecting the right class deltas
    # stacked_roi_indices = tf.stack([index,roi_labels],axis=1)

    # # Getting the RoI deltas for the correct class
    # regression_deltas_for_each_roi_according_to_class = tf.gather_nd(flattened_regression_head,stacked_roi_indices)

    # # Now calculating the loss between the RoI Head and the t* offsets
    # roi_delta_loss = Huber(delta=1.0)
    # deltas_loss = roi_delta_loss(t_star,regression_deltas_for_each_roi_according_to_class)
    

    # return deltas_loss

    # If there is no foreground object
    if tf.shape(t_star)[0] == 0:
        return tf.constant(0.0, dtype=tf.float32)

    # Mask the regression coordinates to remove the padded predictions
    valid_mask = roi_labels > 0
    
    num_pos  = tf.reduce_sum(tf.cast(valid_mask, tf.float32))
    if tf.equal(num_pos, 0.0):
        return tf.constant(0.0, dtype=tf.float32)
        
    valid_regression_head = tf.boolean_mask(regression_head,valid_mask)
    valid_t_star = tf.boolean_mask(t_star,valid_mask)

    # Remove the ignore labels from the Roi Labels using the same mask
    valid_labels = tf.boolean_mask(roi_labels,valid_mask)

    # # Calculate the correct box coordinates from the regression head based on the class
    index = tf.range(valid_regression_head.shape[0],dtype=tf.int32)
    index = tf.stack([index,valid_labels],axis=1)
    
    valid_regression_head = tf.reshape(valid_regression_head,[-1,number_of_classes,4])
    delta_coordinates_per_class = tf.gather_nd(valid_regression_head,index)

    # Now calculating the loss between the RoI Head and the t* offsets
    roi_delta_loss = Huber(delta=1.0,reduction=tf.keras.losses.Reduction.NONE)
    deltas_loss = tf.reduce_sum(roi_delta_loss(valid_t_star,delta_coordinates_per_class))/(num_pos + 1e-8)
    
    return  deltas_loss


# Inference Functions

# Creating a function to convert the bounding boxes from (Xc,Yc,W,H) to (X_MIN,Y_MIN,X_MAX,Y_MAX)
def inference_convert_center_format_boxes_to_xy_coordinate(anchor_boxes, width, height):
    """
    Converts the boxes from the center format of (Xc,Yc,W,H) used by me into the standard format
    (X_MIN,Y_MIN,X_MAX,Y_MAX) used by the model

    Parameters:
    ---------
    anchor_boxes: Tensor
        Tensor of anchor boxes in the format of (Xc,Yc,W,H) 

    Returns:
    -------
    converted_boxes: Tensor
        Tensor of anchor boxes in the converted format of (X_MIN,Y_MIN,X_MAX,Y_MAX)   
    """
    # Creating a utility function for easier conversion in the model since all the anchor boxes made by me are in (X_MIN,Y_MIN,X_MAX,Y_MAX)
    # the model predicts the bounding boxes, its offsets, its errors in the shape of (Xc,Yc,W,H)

    # Split the anchor box into the individual coordinates
    x_c,y_c,w,h = tf.split(anchor_boxes, num_or_size_splits = 4, axis= -1)

    # Calculating the coordinates in XY format
    x_min = x_c - (w/2)
    x_max = x_c + w/2
    y_min = y_c - h/2
    y_max = y_c + h/2

    converted_boxes = tf.concat([x_min,y_min,x_max,y_max],axis=-1) # Stacking the boxes together into one vector
    
    return converted_boxes

# This is a function to apply the bounding boxes deltas to the anchors and the offsets to create the proposals.
def inference_apply_bounding_box_deltas(proposals,deltas,max_width = 800,max_height = 800):
    """
    Apply the deltas to the RoI's for inference run, image wise

    Parameters:
    ---------
    proposals: Tensor
        Tensor of proposals

    deltas: Tensor
        Tensor of all the bounding box deltas

    max_width: Int
        Integer value for the max image width
        
    max_height: Int
        Integer value for the max image height    

    image_stride: Int
        Image Stride to convert Feature map space to Image space
        
    Returns:
    -------
    adjusted_foreground_proposals: Tensor
        Tensor of all the proposals with their offsets added to them
    """
    # Split the proposal coordinates (xc,yc,w,h)
    x_c,y_c,w,h = tf.split(proposals, num_or_size_splits = 4,axis = -1) # Splitting the RoI Coordinates

    t_x,t_y,t_w,t_h = tf.split(deltas, num_or_size_splits = 4,axis = -1) # Splitting the Offsets
    
    # Clipping the t_w and t_h values to prevent too large values that can cause overflow in exp function
    t_w = tf.clip_by_value(t_w,-4,4)
    t_h = tf.clip_by_value(t_h,-4,4)

    x_p = (t_x * w) + x_c
    y_p = (t_y * h) + y_c
    w_p = tf.math.exp(t_w) * w
    h_p = tf.math.exp(t_h) * h

    x1 = tf.clip_by_value(x_p - 0.5 * w_p, 0.0, max_width)
    y1 = tf.clip_by_value(y_p - 0.5 * h_p, 0.0, max_height)
    x2 = tf.clip_by_value(x_p + 0.5 * w_p, 0.0, max_width)
    y2 = tf.clip_by_value(y_p + 0.5 * h_p, 0.0, max_height)

    adjusted_foreground_proposals = tf.concat([x1,y1,x2,y2],axis=-1)
    

    return adjusted_foreground_proposals

# Inference function for the RPN deltas and All the anchors that are generated to create the proposals
def inference_create_proposals(anchors,deltas,image_stride = 16):
    """
    Create Proposals from the RPN by adding the deltas to the anchors
    
    Parameters:
    ---------
    anchors: Tensor
        Tensor of anchors in the shape of [B,50,50,NUM_ANCHORS_PER_PIXEL,4]

    deltas: Tensor
        Tensor of all the bounding box deltas [B,50,50,NUM_ANCHORS_PER_PIXEL,4]

    image_stride: Int
        Image Stride to convert Feature map space to Image space
        
    Returns:
    -------
    stacked_proposal_tensor: Tensor
        Tensor of all the proposals with their offsets added to them
    """
    proposals = []
    for image in range(anchors.shape[0]):
        image_anchors = convert_xy_boxes_to_center_format(anchors[image] * image_stride)
        image_deltas = deltas[image]
        # Create the proposals
        adjusted_proposals = inference_apply_bounding_box_deltas(image_anchors,image_deltas)

        # Append the adjusted proposals to a array to stack them into a tensor
        proposals.append(adjusted_proposals)

    # Now stacking the proposals together
    stacked_proposal_tensor = tf.stack(proposals,axis=0)
        
    return stacked_proposal_tensor

def inference_filter_proposals_by_size(proposals,scores,min_size = 1):
    """
    Filter the proposals by size to remove proposals to reduce the computation going forward

    Parameters:
    ---------
    proposals: Tensor
        Tensor of proposals

    scores: Tensor
        Tensor of all the objectness scores of the Top K proposals [B,K]

    min_size: Int
        Minimum size that is used to be a threshold to filter the proposals
        
    Returns:
    -------
    stacked_filtered_proposals: Tensor
        Stacked filter proposals that are more than the minimum width and minimum height
        
    stacked_filtered_scores: Tensor
        Stacked filtered scores for the corresponding filtered proposals
    """
    filtered_proposals = []
    filtered_scores = []
    for image in range(proposals.shape[0]):
        x_min,y_min,x_max,y_max = tf.split(proposals[image], num_or_size_splits = 4,axis = -1)

        # Calculate the width and height
        height = y_max - y_min
        width = x_max - x_min

        # Use logical AND to get the indices that hold up this condition
        valid_indices = tf.squeeze(tf.where(tf.math.logical_and(tf.squeeze(width) >= min_size,tf.squeeze(height) >= min_size)))

        # Gathering the proposals that are valid
        valid_proposals = tf.gather(proposals[image],valid_indices)

        # Gathering the corresponding scores that are valid
        valid_scores = tf.gather(scores[image],valid_indices)

        filtered_proposals.append(valid_proposals)
        filtered_scores.append(valid_scores)


    # Padding in case the two images don't have the same type of filtering
    num_of_max_proposals = tf.reduce_max([tf.shape(dim)[0] for dim in filtered_proposals])
    padded_tensors = [tf.pad(tensor, paddings = [[0,num_of_max_proposals - tf.shape(tensor)[0]],[0,0]]) for tensor in filtered_proposals]
    padded_score_tensors = [tf.pad(tensor, paddings = [[0,num_of_max_proposals - tf.shape(tensor)[0]]],constant_values=0.0) for tensor in filtered_scores]
    
    stacked_filtered_proposals = tf.stack(padded_tensors,axis=0)
    stacked_filtered_scores = tf.stack(padded_score_tensors,axis=0)
    
    return stacked_filtered_proposals, stacked_filtered_scores

# This is the function to filter the proposals and filtering them to the Top K ones only to reduce the computation
def inference_filter_top_k_proposals(proposals,scores,k_thresh = 300):
    """
    Filter the proposals by Top K

    Parameters:
    ---------
    proposals: Tensor
        Tensor of proposals

    scores: Tensor
        Tensor of all the objectness scores of the Top K proposals [B,K]

    k_thresh: Int
        Filtering size for the proposals to reduce computation
        
    Returns:
    -------
    stacked_filtered_proposals: Tensor
        Stacked filter proposals that are more than the minimum width and minimum height [B,K,4]
        
    stacked_filtered_scores: Tensor
        Stacked filtered scores for the corresponding filtered proposals [B,K]
    """
    # Get the indices of the Top-k scores
    top_k_proposals = []
    top_k_scores = []
    for image in range(proposals.shape[0]):
        flattened_scores = tf.reshape(scores[image],[-1])
          
        # Getting the top k scores and indices
        top_scores,top_indices = tf.math.top_k(flattened_scores,k=k_thresh)

        # Getting the proposals using the indices
        flattened_proposals = tf.reshape(proposals[image],[-1,4])
        top_proposals = tf.gather(flattened_proposals,top_indices)
        
        top_k_proposals.append(top_proposals)
        top_k_scores.append(top_scores)

    # Stack them together
    stacked_top_proposals = tf.stack(top_k_proposals,axis=0)
    stacked_top_scores = tf.stack(top_k_scores,axis=0)

    return stacked_top_proposals, stacked_top_scores

# This is class agnostic and only done to remove the anchors overlapping each other and increasing needless computation.
def inference_apply_class_agnostic_nms_thresholding(proposals,scores,max_output_size = 50,iou_thresholding = 0.5):
    """
    Applying class agnostic NMS thresholding 

    Parameters:
    ---------
    proposals: Tensor
        Tensor of proposals

    scores: Tensor
        Tensor of all the objectness scores of the Top K proposals [B,K]

    max_output_size: Tensor
        Max output size for the NMS function

    iou_thresholding: Float
        IOU threshold for the NMS function
        
    Returns:
    -------
    stacked_proposals: Tensor
        Stacked filter proposals that are more than the minimum width and minimum height [B,NMS,4]
        
    stacked_scores: Tensor
        Stacked filtered scores for the corresponding filtered proposals [B,NMS]
    """
    nms_proposals = []
    nms_scores = []
    for image in range(proposals.shape[0]):
        
        x_min,y_min,x_max,y_max = tf.split(proposals[image], num_or_size_splits = 4,axis = -1)
        
        proposals_stacked = tf.concat([y_min,x_min,y_max,x_max] ,axis=-1)
        
        selected_indices = tf.image.non_max_suppression(proposals_stacked,scores[image],max_output_size=max_output_size,iou_threshold = iou_thresholding)

        selected_proposals = tf.gather(proposals[image],selected_indices)

        selected_scores = tf.gather(scores[image],selected_indices)

        nms_proposals.append(selected_proposals)
        nms_scores.append(selected_scores)

    num_of_max_proposals = tf.reduce_max([tf.shape(dim)[0] for dim in nms_proposals])
    dummy_box = tf.constant([[0,0,1e-3,1e-3]],dtype=nms_proposals[0].dtype)
    padded_tensors = [tf.concat([tensor,tf.tile(dummy_box,multiples=[num_of_max_proposals - tf.shape(tensor)[0],1])],axis=0) for tensor in nms_proposals]
    padded_score_tensors = [tf.pad(tensor, paddings = [[0,num_of_max_proposals - tf.shape(tensor)[0]]],constant_values=-1e-9) for tensor in nms_scores]
    
    stacked_proposals = tf.stack(padded_tensors,axis = 0)
    stacked_scores = tf.stack(padded_score_tensors,axis=0)

    return stacked_proposals, stacked_scores

# This is the overarching proposal filtering stage that calls all the smaller steps
def inference_filter_proposals(proposals,objectness_score,nms_output_size = 50,nms_iou_threshold = 0.7):
    """
    Inference function for filtering proposals from the RPN to creating the proposals all the way to the RoI pooling

    Parameters:
    ---------
    proposals: Tensor
        Tensor of proposals

    objectness_score: Tensor
        Tensor of all the objectness scores of the Top K proposals [B,K]

    nms_output_size: Tensor
        Max output size for the NMS function

    nms_iou_threshold: Float
        IOU threshold for the NMS function
        
    Returns:
    -------
    nms_proposals: Tensor
        Stacked filter proposals that are more than the minimum width and minimum height [B,NMS,4]
        
    nms_scores: Tensor
        Stacked filtered scores for the corresponding filtered proposals [B,NMS]
    """
    # Select Top K proposals
    top_proposals,top_scores = inference_filter_top_k_proposals(proposals,objectness_score)
     # Filter the small anchor boxes to remove unnecssary anchors
    top_proposals,top_scores = inference_filter_proposals_by_size(top_proposals,top_scores,min_size=1)
    # Apply NMS
    nms_proposals, nms_scores = inference_apply_class_agnostic_nms_thresholding(top_proposals,top_scores,max_output_size = nms_output_size ,iou_thresholding=nms_iou_threshold)
    
    return nms_proposals, nms_scores

# This is the implementation of the RoI Align stage in the Inference Pipeline
def inference_roi_align(feature_map,proposals,scores,output_size = (7,7),image_size=(800,800)):
    """
    RoI Align
    
    Parameters:
    ---------
    feature_map: Tensor
        Feature Map from the RPN (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,CHANNELS)

    proposals : Tensor
        Tensor of inference proposal boxes in Image Space

    roi_batch_indices : Tensor
        Batch indices for RoI's

    output_size: (int,int)
        Output grid RoI size (Row,Col)

    image_size: (int,int)
        Image Size Tuple (Row,Col)
    Returns:
    -------
    inference_rois: Tensor
        RoI's in the cropped and resized shape of (B,NUM_ROI,GRID_ROW_SIZE,GRID_COL_SIZE,CHANNELS)
    """
    rois = []
    for image in range(proposals.shape[0]):
        # Filter out the padded 
        indices = tf.where(scores[image] != -1e-9)

        proposals_image = tf.gather_nd(proposals[image],indices)
        # Normalizing the box coordinates
        x_min,y_min,x_max,y_max = tf.split(proposals_image,num_or_size_splits = 4,axis=-1)
    
        x_min_normalized = x_min/image_size[1]
        y_min_normalized = y_min/image_size[0]
        x_max_normalized = x_max/image_size[1]
        y_max_normalized = y_max/image_size[0]

        # Stacking the coordinates together
        normalized_boxes = tf.concat([y_min_normalized, x_min_normalized, y_max_normalized, x_max_normalized],axis=-1)

        inference_rois = tf.image.crop_and_resize(feature_map,normalized_boxes,tf.repeat([image],repeats=[indices.shape[0]]),output_size)

        rois.append(inference_rois)
        
    # Padding the tensors so that they can be stacked
    num_of_max_rois = tf.reduce_max([tf.shape(dim)[0] for dim in rois])
    padded_tensors = [tf.pad(tensor, paddings = [[0,num_of_max_rois - tf.shape(tensor)[0]],[0,0],[0,0],[0,0]]) for tensor in rois]

    # Stacking the two tensors together
    stacked_roi_tensors = tf.stack(padded_tensors,axis=0)

    return stacked_roi_tensors

# Inference Function for the RoI Head to get the valid class probabilites
def inference_classification_decision(classification_scores,valid_mask):
    """
    Inference classification decision to get the prediction labels and their prediction scores for each RoI in an image

    Parameters:
    ---------
    classification_scores : Tensor
        Tensor of classification scores from the RoI Head (B,NUM_ROIS,NUM_CLASSES)

    valid_mask: Tensor
        Tensor of valid mask used to find the padded scores and proposal indices

    Returns:
    -------
    class_probabilities: Tensor
        Prediction foreground scores for the RoIs in the image in the shape of (B,NUM_ROIS,FG_CLASS)
    """
    # Mapped the probabilties using softmax
    class_probabilities = tf.nn.softmax(classification_scores, axis = -1)

    # Substituting the padded boxes scores with sentinel scores to make it easier to work with
    class_probabilities = tf.where(valid_mask[...,tf.newaxis],class_probabilities,tf.zeros_like(class_probabilities))
    
    return class_probabilities[:,:,1:]

# Inference function to apply deltas to the RoI coordinates
def inference_apply_deltas(roi_coordinates,bbox_deltas,number_of_classes,image_width = 800, image_height = 800):
    """
    Apply the deltas to the RoI's for inference run

    Parameters:
    ---------
    roi_coordinates: Tensor
        Tensor of RoI coordinates in Image Space

    bbox_deltas: Tensor
        Tensor of all the bounding box deltas

    pred_labels: Tensor
        Prediction labels for the RoIs in the image in the shape of (B,NUM_ROIS)

    number_of_classes: Int
        Number of classes in the RoI Head
    Returns:
    -------
    stacked_inference_tensor: Tensor
        Tensor of all the proposals with their offsets added to them (B,NUM_FG,4) In Image Space
    """
    # # Gather the correct class bounding box deltas
    # inference_rois = []
    # for image in range(pred_labels.shape[0]):
    #     # Gather deltas
    #     deltas_per_class = tf.reshape(bbox_deltas[image],[-1,number_of_classes,4])
    #     roi_selection_index = tf.range(pred_labels[image].shape[0],dtype=tf.int64)
    #     roi_indices = tf.stack([roi_selection_index,pred_labels[image]],axis=1)
    #     roi_deltas = tf.gather_nd(deltas_per_class,roi_indices)
    #     center_roi_coordinates = convert_xy_boxes_to_center_format(roi_coordinates[image])
    #     # Add deltas
    #     adjusted_rois = inference_apply_bounding_box_deltas(center_roi_coordinates,roi_deltas)

    #     # Append it to the inference RoI's
    #     inference_rois.append(adjusted_rois)

    # # Stacking the tensors together
    # stacked_inference_tensor = tf.stack(inference_rois, axis=0)
    
    # return stacked_inference_tensor

    bbox_deltas_per_class = tf.reshape(bbox_deltas,[-1,bbox_deltas.shape[1],number_of_classes,4])
    bbox_deltas_per_fg_class = bbox_deltas_per_class[:,:,1:,:]

    x1,y1,x2,y2 =  tf.split(roi_coordinates,num_or_size_splits = 4,axis=-1)

    w = tf.maximum(x2-x1,1e-6)
    h = tf.maximum(y2-y1,1e-6)
    cx = x1 + (0.5*w)
    cy = y1 + (0.5*h)

    tx,ty,tw,th = tf.split(bbox_deltas_per_fg_class,num_or_size_splits = 4,axis=-1)
    
    tw = tf.clip_by_value(tw,-4,4)
    th = tf.clip_by_value(th,-4,4)

    x_p = (tx * w[...,tf.newaxis]) + cx[...,tf.newaxis]
    y_p = (ty * h[...,tf.newaxis]) + cy[...,tf.newaxis]
    w_p = tf.exp(tw) * w[...,tf.newaxis]
    h_p = tf.exp(th) * h[...,tf.newaxis]

    x_min = tf.maximum(x_p - 0.5*w_p,0)
    y_min = tf.maximum(y_p - 0.5*h_p,0)
    x_max = tf.minimum(x_p + 0.5*w_p,image_width)
    y_max = tf.minimum(y_p + 0.5*h_p,image_height)

    stacked_inference_tensor = tf.stack([x_min,y_min,x_max,y_max],axis=-1)

    return tf.squeeze(stacked_inference_tensor,axis= -2)

# This is the function to carry out combined non max suppression for the inference pipeline to get the final boxes, scores, classes
def inference_combined_non_max_suppression(proposals,scores,scores_thresh,iou_thresh,max_total_size,max_output_size_per_class,image_width = 800,image_height = 800):
    """
    Inference combined non max suppression to get the final boxes, scores, labels, valid detection numbers

    Parameters:
    ---------
    proposals: Tensor
        Tensor of RoI proposals

    scores: Tensor
        Tensor of predicted scores from the RoI Head

    scores_thresh: Float
        Scores threshold for the Combined NMS

    iou_thresh: Float
        IoU thresh for the Combined NMS to filter boxes

    max_total_size: Int
        Max Total Predictions for the Combined NMS

    max_output_size_per_class: Int
        Max output size per class to be selected by the NMS per class of the model
    Returns:
    -------
    nms_boxes: Tensor
        Tensor of filtered final boxes using NMS in the shape (B,MAX_PER_CLASS,4) In Image Space

    nms_scores: Tensor
        Tensor of filtered final probabilites for the boxes using NMS in the shape (B,MAX_PER_CLASS)

    nms_classes: Tensor
        Tensor of filtered final labels for the boxes using NMS in the shape (B,MAX_PER_CLASS)

    valid_detections: Tensor
        Tensor of valid detections in each done by NMS in the shape (B)
    """
    # Making them in the format by first normalizing
    x_min,y_min,x_max,y_max = tf.split(proposals,num_or_size_splits = 4,axis=-1)

    # Normalizing them to [0,1]
    x_min = x_min / image_width
    y_min = y_min / image_height
    x_max = x_max / image_width
    y_max = y_max / image_height

    # Rearranging them into the format of [y1,x1,y2,x2]
    normalized_boxes = tf.concat([y_min,x_min,y_max,x_max], axis = -1)

    # Passing it through the NMS
    nms_boxes, nms_scores, nms_classes,valid_detections = tf.image.combined_non_max_suppression(normalized_boxes,scores,max_output_size_per_class = max_output_size_per_class,max_total_size = max_total_size,iou_threshold = iou_thresh,score_threshold = scores_thresh)

    return nms_boxes, nms_scores, nms_classes,valid_detections

# This is the helper function to get the colour palette for the inference box to be displayed over the image
def inference_get_colour_palette(cls_id):
    """
    Get the colour palette for the class id from the predefined options

    Parameters:
    ---------
    cls_id: int
        Class id for the class

    Returns:
    -------
    palette: Tuple [Float, Float, Float]
        Color of the prediction to display on the image during inference
    """
    palette = [
            (0.95, 0.35, 0.35), (0.35, 0.55, 0.95), (0.35, 0.85, 0.55),
            (0.95, 0.75, 0.35), (0.75, 0.35, 0.95), (0.35, 0.85, 0.85),
            (0.85, 0.35, 0.55), (0.55, 0.35, 0.85), (0.55, 0.75, 0.35),
        ]
    return palette[int(cls_id) % len(palette)]

# This is the function to get the formatted label text for the predicted box
def inference_get_label_text(labels,scores,id_,class_names,offset = 0):
    """
    Get the label text for the bounding box

    Parameters:
    ---------
    labels: Array [String]
        Tensor of RoI proposals

    scores: Array [Float]
        Class scores for the box

    id_: int
        Class id for the class

    class_names: Array[String]

    Returns:
    -------
    name: String
        Label name for the bounding box
    """
    
    # Getting the class id using the offset. If it is a prediction then the offset is 0 since there is no background class
    # Else if it is a ground truth box then the offset is 1 since there is a background class in the RoI Head which is '0'
    cls_id = int(labels[id_]) 
    name = None
    if class_names is not None:
        if isinstance(class_names, dict):
            name = class_names.get(cls_id, str(cls_id))
        else:
            lookup_id = cls_id + offset
            # if class ids are 1..K, shift; if 0..K-1, use directly
            if 0 <= lookup_id < len(class_names):
                name = class_names[lookup_id]
    name = name if name is not None else str(cls_id)
    if scores is not None:
        return f"{name}: {scores[id_]:.2f}"
    return name

# Inference functon to plot the bounding boxes for the image
def inference_plot_predicted_bounding_boxes(image,nms_boxes,nms_scores,nms_classes,class_names,order='xyxy',is_normalized = False,gt_boxes = None, gt_classes = None, gt_order = 'xyxy', gt_is_normalized = False, show_legend = False, debug=False):
    """
    Plot the predicted boundobg boxes on top of the image to judge the inference.

    Parameters:
    ---------
    image: Tensor
        Tensor of Image [800,800,3]

    nms_boxes: Tensor
        Tensor of NMS boxes [NMS,4] for the image

    nms_classes: Tensor
        Tensor of NMS classes [NMS] for the image

    class_names: Array[String]
        List of Class Names for the dataset

    order: String
        String used to see if the boxes need to be converted into xyxy

    is_normalized: Bool
        Boolean flag used to check if the boxes are normalizes and if so then denormalizing them before displaying them

    debug: Bool
        Boolean flag to view debug information from the function
    Returns:
    -------
    palette: Tuple [Float, Float, Float]

    """
    img = np.asarray(image)
    H,W = img.shape[:2]
    boxes = np.asarray(nms_boxes)
    labels = np.asarray(nms_classes)

    if nms_scores is not None:
        scores = np.asarray(nms_scores)

    # Allowing for robust changing
    if order.lower() == 'yxyx':
        y1,x1,y2,x2 = np.split(boxes,4,axis=-1)
        boxes = np.concatenate([x1,y1,x2,y2],axis=-1)
        

    # Denormalizing the boxes if necessary
    if is_normalized:
        boxes = boxes.copy()
        boxes[:,[0,2]] = boxes[:,[0,2]] * W
        boxes[:,[1,3]] = boxes[:,[1,3]] * H

    # Clip the image to W-1 and H-1 for viewing QoL
    boxes[:,[0,2]] = np.clip(boxes[:,[0,2]], 0, W-1)
    boxes[:,[1,3]] = np.clip(boxes[:,[1,3]], 0, H-1)
    
    # Checking for ground truth boxes
    gt_present = gt_boxes is not None and gt_classes is not None
    if gt_present:
        # Check for the order
        gt_boxes = np.asarray(gt_boxes).copy()
        if gt_order.lower() == 'yxyx':
            y1,x1,y2,x2 = np.split(gt_boxes,4,axis=-1)
            gt_boxes = np.concatenate([x1,y1,x2,y2],axis=-1)
            
        # Denormalizing the boxes if necessary
        if gt_is_normalized:
            gt_boxes = gt_boxes.copy()
            gt_boxes[:,[0,2]] = gt_boxes[:,[0,2]] * W
            gt_boxes[:,[1,3]] = gt_boxes[:,[1,3]] * H
            
        # Clip the image to W-1 and H-1 for viewing QoL
        x1 = tf.clip_by_value(gt_boxes[:, 0], 0.0, float(W - 1))
        y1 = tf.clip_by_value(gt_boxes[:, 1], 0.0, float(H - 1))
        x2 = tf.clip_by_value(gt_boxes[:, 2], 0.0, float(W - 1))
        y2 = tf.clip_by_value(gt_boxes[:, 3], 0.0, float(H - 1))
        gt_boxes = tf.stack([x1, y1, x2, y2], axis=1)
        gt_labels = np.asarray(gt_classes)

    # Create the figure
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    ax.imshow(img)
    ax.axis("off")

    # Loop through the info
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
        if w <= 0 or h <= 0:
            if debug:
                print(f'Box: {x1},{y1},{x2},{y2} is degenerate')
            continue  # skip degenerate
        color = inference_get_colour_palette(labels[i])
        if debug:
            print(f'Box: {x1},{y1},{x2},{y2}')

        # rectangle
        rect = patches.Rectangle(
            (x1, y1), w, h, linewidth=1, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

        # label box
        txt = inference_get_label_text(labels,scores,i,class_names=class_names,offset=0)
        ax.text(
            x1, y1 - 2, txt,
            fontsize=10, color="white",
            bbox=dict(facecolor=color, alpha=1.0, edgecolor='none', pad=2)
        )
        
        if gt_present:
            # Plotting the ground truth boxes
            for j in range(len(gt_boxes)):
                gx1, gy1, gx2, gy2 = map(float, gt_boxes[j])  # Ensure float type
                gw, gh = max(0.0, gx2 - gx1), max(0.0, gy2 - gy1)
                if gw <= 0 or gh <= 0:
                    if debug:
                        print(f'GT Box: {gx1},{gy1},{gx2},{gy2} is degenerate')
                    continue  # skip degenerate
                
                gt_color = (0.0, 1.0, 0.0)  # Green for ground truth
                if debug:
                    print(f'GT Box: {gx1},{gy1},{gx2},{gy2}')

                # rectangle
                gt_rect = patches.Rectangle(
                    (gx1, gy1), gw, gh, linewidth=1, edgecolor=gt_color, facecolor='none', linestyle='--'
                )
                ax.add_patch(gt_rect)

                # label box
                gt_txt = inference_get_label_text(gt_labels,None,j,class_names=class_names,offset=-1)
                ax.text(
                    gx1, gy1 - 2, gt_txt,
                    fontsize=10, color="white",
                    bbox=dict(facecolor=gt_color, alpha=1.0, edgecolor='none', pad=2)
                )
                
        if show_legend:
            # Showing the legend
            pred_line = patches.Patch(edgecolor='k', facecolor='none', label='Prediction')
            gt_line = patches.Patch(edgecolor='k', facecolor='none', label='Ground Truth')
            
            ax.legend(handles=[pred_line, gt_line], loc='lower right', frameon=True)

    plt.tight_layout()
    return fig, ax
        
## Utility Functions for Metrics

def util_iou_xyxy(a,b):
    """
    Util Function to calculate IoU for the Utility functions in the metrics for the model

    Parameters:
    ---------
    a : Tensor
        Tensor Of IOU Scores Shape (B,N,4)
        
    b : Tensor
        Tensor Of IOU Scores Shape (M,4)

    Returns:
    -------
    ious: Tensor
        Tensor Of IoU Scores (B, N, M)
    """
    
    x11,y11,x21,y21 = tf.split(a,num_or_size_splits=4,axis=-1)
    x12,y12,x22,y22 = tf.split(b,num_or_size_splits=4,axis=-1)
    
    x1_max = tf.maximum(x11,tf.transpose(x12, [0,2,1]))
    y1_max = tf.maximum(y11,tf.transpose(y12, [0,2,1]))
    x2_min = tf.minimum(x21,tf.transpose(x22, [0,2,1]))
    y2_min = tf.minimum(y21,tf.transpose(y22, [0,2,1]))

    width = tf.maximum(0.0,x2_min - x1_max)
    height = tf.maximum(0.0,y2_min - y1_max)

    intersection = width * height

    area_a = (x21 - x11) * (y21 - y11)
    area_b = (x22 - x12) * (y22 - y12)

    area_b = tf.transpose(area_b,[0,2,1])

    union = area_a + area_b - intersection + 1e-8 
    
    return intersection/union


def util_rpn_recall_post_nms(proposals_xyxy,ground_truth_boxes_xyxy,gt_labels,iou_thresh = 0.5):
    """
    Util Function to calculate recall for the RPN

    Parameters:
    ---------
    proposals_xyxy : Tensor
        Tensor of NMS proposals (B,NMS_PROPS,4)
        
    ground_truth_boxes_xyxy : Tensor
        Tensor of Ground Truth boxes (B,NUM_GT,4)

    gt_labels : Tensor
        Tensor of Ground Truth boxes (B,NUM_GT)

    iou_thresh : Float
        Threshold for IoU threshold to calculate the recall
        
    Returns:
    -------
    per_img: Tensor
        Tensor Of IoU Scores (B,)

    global_recall: Tensor
        Tensor Of Global recall (1,)
    """
    gt_mask = gt_labels >= 1

    if ground_truth_boxes_xyxy.shape[1] == 0:
        return tf.constant(1.0)

    iou = util_iou_xyxy(proposals_xyxy,ground_truth_boxes_xyxy)
    covered_gt = tf.reduce_any(iou >= iou_thresh,axis=1) & gt_mask

    gt_valid = tf.cast(gt_mask,tf.float32)
    
    num_cov = tf.reduce_sum(tf.cast(covered_gt, tf.float32), axis=1)
    num_gt  = tf.reduce_sum(tf.cast(gt_mask, tf.float32), axis=1)

    per_img = tf.math.divide_no_nan(num_cov, num_gt)

    global_recall = tf.math.divide_no_nan(tf.reduce_sum(num_cov), tf.reduce_sum(num_gt))
    
    return per_img, global_recall

def util_roi_sampler_health(roi_labels,pos_label_min = 1,sample_target = 512,pos_target = 0.25):
    """
    Util Function to see the health for the RoI Sampler

    Parameters:
    ---------
    roi_labels : Tensor
        Tensor of NMS proposals (B,NUM_SAMPLE)
        
    pos_label_min : Int
        Label value that is used to distinguish the foreground from the background

    sample_target : Int
        The total number of RoI's that are sampled by the sampler

    pos_target : Float
        The ratio for the ratio of positive RoI's that need to be present in the sample
        
    Returns:
    -------
    per_img: Tensor
        Tensor Of IoU Scores (B,)

    global_recall: Tensor
        Tensor Of Global recall (1,)
    """
    # Get the total number of positive boxes in the RoI's
    pos_mask = roi_labels >= pos_label_min
    
    pos = tf.reduce_sum(tf.cast(pos_mask,tf.float32),axis=1)
    pos_frac = tf.cast(pos,tf.float32) / tf.cast(roi_labels.shape[1],tf.float32)
    
    return {
        "roi_total": roi_labels.shape[1],
        "roi_pos_frac_per_image": pos_frac.numpy(),
        "roi_pos_frac": tf.reduce_mean(tf.cast(pos_frac,tf.float32)).numpy(),
        "target_range_deviation": (tf.abs(tf.cast(roi_labels.shape[1], tf.int32) - sample_target) <= 16).numpy(),
        "pos_frac_deviation": (tf.abs(pos_frac - pos_target)).numpy()
    }
    
def util_calculate_corresponding_iou_scores(proposals,gt_boxes):
    """
    Calculates IoU for the proposals for the RoI Head regressor health
    
    Parameters:
    ---------
    proposals: Tensor
        Tensor of proposals in the shape of (B,NUM_SAMPLE,4)

    gt_boxes: Tensor
        Tensor of proposals in the shape of (B,NUM_GT_BOXES,4)
    
    Returns:
    -------
    iou: Tensor
        Tensor of IoU scores in the shape (B,NUM_SAMPLE,NUM_GT_BOXES)
    """
    p_xmin,p_ymin,p_xmax,p_ymax = tf.split(proposals,num_or_size_splits = 4,axis = -1)
    gt_xmin,gt_ymin,gt_xmax,gt_ymax = tf.split(gt_boxes,num_or_size_splits = 4,axis = -1)
    
    x1_max = tf.maximum(p_xmin,gt_xmin)
    y1_max = tf.maximum(p_ymin,gt_ymin)
    x2_min = tf.minimum(p_xmax,gt_xmax)
    y2_min = tf.minimum(p_ymax,gt_ymax)

    width = tf.maximum(0.0,x2_min - x1_max)
    height = tf.maximum(0.0,y2_min - y1_max)

    intersection = width * height

    area_a = (p_xmax - p_xmin) * (p_ymax - p_ymin)
    area_b = (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin)

    union = area_a + area_b - intersection + 1e-8 
    
    return tf.squeeze(intersection/union,axis=-1)

def util_add_deltas_to_proposals(proposals,deltas,max_width = 800, max_height = 800):
    """
    Adds deltas to the proposals to the proposals
    
    Parameters:
    ---------
    proposals: Tensor
        Tensor of proposals in the shape of (B,NUM_POS,4) in xy-coordinate system

    deltas: Tensor
        Tensor of deltas in the shape of (B,NUM_POS,4)
    
    Returns:
    -------
    proposals: Tensor
        Tensor of proposals with the offsets added (B,NUM_POS,4) in xy-coordinate system
    """
    
    x_c,y_c,w,h = tf.split(proposals, num_or_size_splits = 4,axis = -1) # Splitting the RoI Coordinates

    t_x,t_y,t_w,t_h = tf.split(deltas, num_or_size_splits = 4,axis = -1) # Splitting the Offsets

    x_p = (t_x * w) + x_c
    y_p = (t_y * h) + y_c
    w_p = tf.math.exp(t_w) * w
    h_p = tf.math.exp(t_h) * h

    x1 = tf.clip_by_value(x_p - 0.5 * w_p, 0.0, max_width)
    y1 = tf.clip_by_value(y_p - 0.5 * h_p, 0.0, max_height)
    x2 = tf.clip_by_value(x_p + 0.5 * w_p, 0.0, max_width)
    y2 = tf.clip_by_value(y_p + 0.5 * h_p, 0.0, max_height)

    proposals = tf.concat([x1,y1,x2,y2],axis=-1)
    
    return proposals
    
def util_roi_head_regressor_gain(roi_proposals,roi_labels,matched_gt_boxes,roi_deltas,ground_truth_boxes,number_of_classes):
    """
    Utility function to calculate the regressor gain
    
    Parameters:
    ---------
    roi_proposals: Tensor
        Tensor of proposals in the shape of (B,NUM_SAMPLE,4)

    roi_labels: Tensor
        Tensor of RoI labels in the shape of (B,NUM_SAMPLE)

    matched_gt_boxes: Tensor
        Tensor of deltas in the shape of (B,NUM_SAMPLE,4)

    roi_deltas: Tensor
        Tensor of deltas in the shape of (B,NUM_SAMPLE,NUM_CLASSES * 4)

    ground_truth_boxes: Tensor
        Tensor of ground truth boxes in the shape of (B,NUM_GT_BOXES,4)

    number_of_classes: Int
        Number of classes in the RoI Head 
    
    Returns:
    -------
    iou_gain: Tensor
        IoU gain for the RoI Head Regressor
    """
    
    # First gather all the foreground boxes and their matched gt boxes
    pos_mask = roi_labels > 0
    pos_rois = tf.boolean_mask(roi_proposals,pos_mask)
    pos_gt_boxes = tf.boolean_mask(matched_gt_boxes,pos_mask)

    # Now use them to calculate the IOU values between them for evaluation

    iou_before = util_calculate_corresponding_iou_scores(pos_rois,pos_gt_boxes)

    # Gather the labels that are for the labels
    pos_labels = tf.boolean_mask(roi_labels,pos_mask)
    pos_deltas = tf.boolean_mask(tf.reshape(roi_deltas,[-1,roi_deltas.shape[1],number_of_classes,4]),pos_mask)
    pos_deltas = tf.gather(pos_deltas,pos_labels,axis=1,batch_dims=1)

    center_proposals = convert_xy_boxes_to_center_format(pos_rois)
    refined_proposals = util_add_deltas_to_proposals(center_proposals,pos_deltas)

    # Calculate IoU after

    iou_after = util_calculate_corresponding_iou_scores(refined_proposals,pos_gt_boxes)

    # Calculate IoU gain

    iou_gain = iou_after - iou_before
    
    return tf.math.reduce_mean(iou_gain)

def iou_xyxy(a, b):
    """
    Util Function to calculate IoU for the Utility functions in the mAP metric

    Parameters:
    ---------
    a : Tensor
        Tensor Of IOU Scores Shape (B,N,4)
        
    b : Tensor
        Tensor Of IOU Scores Shape (M,4)

    Returns:
    -------
    ious: Tensor
        Tensor Of IoU Scores (B, N, M)
    """
    # a: (4,), b: (K,4)
    x1 = np.maximum(a[0], b[:,0]); y1 = np.maximum(a[1], b[:,1])
    x2 = np.minimum(a[2], b[:,2]); y2 = np.minimum(a[3], b[:,3])
    w = np.maximum(0.0, x2-x1); h = np.maximum(0.0, y2-y1)
    inter = w*h
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[:,2]-b[:,0])*(b[:,3]-b[:,1]) - inter + 1e-9
    return inter/ua

def voc_ap(rec, prec, use_11pt=False):
    """
   Calculating the AP for the model on the VOC dataset

    Parameters:
    ---------
    rec : Tensor
        Recall Scores for the images
        
    prec : Tensor
        Precision Scores for the images

    Returns:
    -------
    ap: Tensor
        Average precision of the model
    """
    if use_11pt:
        ap = 0.0
        for t in np.linspace(0,1,11):
            p = prec[rec>=t].max() if np.any(rec>=t) else 0.0
            ap += p/11.0
        return ap
    # continuous (VOC 2010): precision envelope
    mrec  = np.concatenate(([0.], rec, [1.]))
    mprec = np.concatenate(([0.], prec, [0.]))
    for i in range(mprec.size-2, -1, -1):
        mprec[i] = max(mprec[i], mprec[i+1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[idx+1]-mrec[idx]) * mprec[idx+1])

def compute_map_voc(
    det_boxes,    # list of (Ni,4) arrays, xyxy per image
    det_scores,   # list of (Ni,) arrays
    det_labels,   # list of (Ni,) int arrays in 1..C
    gt_boxes,     # list of (Mi,4) arrays, xyxy per image
    gt_labels,    # list of (Mi,) int arrays in 1..C
    num_classes,  # C
    iou_thr=0.5,
    use_11pt=False
):
    """
    Calculating the AP for the model on the VOC dataset

    Parameters:
    ---------
    rec : Tensor
        Recall Scores for the images
        
    prec : Tensor
        Precision Scores for the images

    Returns:
    -------
    mAP: float
        mAP for the model

    ap_per_class: np.ndarray shape (C,) 
        AP per class with NaN for classes with no GT
    """
    assert len(det_boxes)==len(det_scores)==len(det_labels)==len(gt_boxes)==len(gt_labels)
    n_images = len(det_boxes)

    # --- build per-class structures ---
    # gts[c]: dict img_id -> {'boxes': ndarray (G,4), 'matched': np.zeros(G,bool)}
    gts = [ {} for _ in range(num_classes) ]
    npos = np.zeros(num_classes, dtype=int)
    # preds[c]: list of (score, img_id, box[4])
    preds = [ [] for _ in range(num_classes) ]

    for img_id in range(n_images):
        # GT per class
        gl = gt_labels[img_id]
        gb = gt_boxes[img_id]
        for c in range(1, num_classes+1):
            mask = (gl == c)
            if np.any(mask):
                boxes_c = gb[mask]
                gts[c-1][img_id] = {'boxes': boxes_c, 'matched': np.zeros(len(boxes_c), dtype=bool)}
                npos[c-1] += boxes_c.shape[0]
        # Detections per class
        dl = det_labels[img_id]
        db = det_boxes[img_id]
        ds = det_scores[img_id]
        for i in range(len(dl)):
            c = int(dl[i])
            if 1 <= c <= num_classes:
                preds[c-1].append((float(ds[i]), img_id, db[i].astype(float)))


    ap_per_class = np.full(num_classes, np.nan, dtype=float)

    for c in range(num_classes):
        if npos[c] == 0:
            continue  # no GT for this class → AP undefined (kept NaN, excluded from mAP)

        dets = preds[c]
        if len(dets) == 0:
            ap_per_class[c] = 0.0
            continue

        # sort detections by score desc
        dets.sort(key=lambda t: -t[0])

        tp = np.zeros(len(dets), dtype=float)
        fp = np.zeros(len(dets), dtype=float)

        for i, (_, img_id, box) in enumerate(dets):
            # get GTs of this class in this image
            g = gts[c].get(img_id, None)
            if g is None or g['boxes'].shape[0] == 0:
                fp[i] = 1.0
                continue

            ious = iou_xyxy(box, g['boxes'])
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr and not g['matched'][j]:
                tp[i] = 1.0
                g['matched'][j] = True
            else:
                fp[i] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / (npos[c] + 1e-9)
        prec = tp_cum / (tp_cum + fp_cum + 1e-9)

        ap_per_class[c] = voc_ap(rec, prec)

    
    valid = ~np.isnan(ap_per_class)
    mAP = float(np.mean(ap_per_class[valid])) if np.any(valid) else 0.0
    return mAP, ap_per_class

def evaluation_compute_mAP(
    det_boxes_b,   # [B, K, 4]
    det_scores_b,  # [B, K]
    det_labels_b,  # [B, K] ints in 1..C (0/-1 for pad)
    gt_boxes_b,    # [B, G, 4]
    gt_labels_b,   # [B, G]  ints in 1..C (0/-1 for pad)
    num_classes,
    iou_thr=0.5,
    use_11pt=False,
    det_mask_b=None,  # optional [B, K] bool
    gt_mask_b=None    # optional [B, G] bool
):
    """
    Calculating the AP for the model on the VOC dataset

    Parameters:
    ---------
    det_boxes_b: Tensor
        Batched detection boxes in the Shape (B,K,4)
        
    det_scores_b: Tensor
        Batched detection scores in the Shape (B,K)

    det_labels_b: Tensor
        Batched detection labels in the Shape (B,K)

    gt_boxes_b: Tensor
        Tensor for the ground truth boxes in the shape of (B,NUM_GT_BOXES,4)

    gt_labels_b: Tensor
        Tensor for the ground truth boxes in the shape of (B,NUM_GT_BOXES)

    num_classes: Int
        The number of classes in the dataset

    iou_thr: Float
        The threshold of the IoU to calculate the mAP for the model
        
    Returns:
    -------
    mAP: Float
        mAP for the model

    ap_per_class: np.ndarray shape (C,) 
        AP per class with NaN for classes with no GT
    """
    # Convert TF tensors to np arrays if needed
    det_boxes_b  = np.asarray(det_boxes_b)
    det_scores_b = np.asarray(det_scores_b)
    det_labels_b = np.asarray(det_labels_b)
    gt_boxes_b   = np.asarray(gt_boxes_b)
    gt_labels_b  = np.asarray(gt_labels_b)

    B, K = det_labels_b.shape
    G    = gt_labels_b.shape[1]

    # Default masks if not provided
    if det_mask_b is None:
        det_mask_b = (det_labels_b > 0)  # or (det_scores_b > 0)
    else:
        det_mask_b = np.asarray(det_mask_b)
    if gt_mask_b is None:
        gt_mask_b = (gt_labels_b > 0)
    else:
        gt_mask_b = np.asarray(gt_mask_b)

    # Build per-image lists expected by compute_map_voc
    det_boxes_list, det_scores_list, det_labels_list = [], [], []
    gt_boxes_list,  gt_labels_list  = [], []

    for b in range(B):
        dm = det_mask_b[b]
        gm = gt_mask_b[b]

        # detections
        if dm.any():
            det_boxes_list.append(det_boxes_b[b][dm])
            det_scores_list.append(det_scores_b[b][dm])
            det_labels_list.append(det_labels_b[b][dm])
        else:
            det_boxes_list.append(np.zeros((0,4), dtype=float))
            det_scores_list.append(np.zeros((0,),  dtype=float))
            det_labels_list.append(np.zeros((0,),  dtype=int))

        # ground truth
        if gm.any():
            gt_boxes_list.append(gt_boxes_b[b][gm])
            gt_labels_list.append(gt_labels_b[b][gm])
        else:
            gt_boxes_list.append(np.zeros((0,4), dtype=float))
            gt_labels_list.append(np.zeros((0,),  dtype=int))

    # Call the per-image evaluator from before
    mAP, ap_per_class = compute_map_voc(
        det_boxes_list, det_scores_list, det_labels_list,
        gt_boxes_list, gt_labels_list,
        num_classes=num_classes, iou_thr=iou_thr, use_11pt=use_11pt
    )
    return mAP, ap_per_class

def util_percentile(x, q, axis=None):
    """
    Percentile along axis using tf.sort (no TFP).
    x: Tensor, q in [0,100]
    """
    x = tf.convert_to_tensor(x)
    q = tf.convert_to_tensor(q, dtype=tf.float32)
    if axis is None:
        x = tf.reshape(x, [-1])
        axis = 0
    k = tf.cast(tf.shape(x)[axis], tf.float32)
    rank = tf.clip_by_value(tf.cast(tf.math.floor((q/100.0)*(k-1.0)), tf.int32), 0, tf.shape(x)[axis]-1)
    x_sorted = tf.sort(x, axis=axis)
    return tf.gather(x_sorted, rank, axis=axis)

def util_roi_head_p95_non_bg(cls_scores, from_logits=True):
    """
    For RoI head scores (B, N, C), return the 95th percentile of the max foreground
    confidence per image. Useful to track if the head is 'confident enough' on FG.
    """
    s = cls_scores
    if from_logits:
        s = tf.nn.softmax(s, axis=-1)
    # foreground = classes 1..C-1
    fg_probs = s[:, :, 1:]
    conf = tf.reduce_max(fg_probs, axis=-1)  # (B, N)
    # per-image p95
    p95 = tf.map_fn(lambda v: util_percentile(v, 95.0, axis=0), conf, fn_output_signature=conf.dtype)
    return p95  # shape (B,)

def evaluation_prf1_report(
    det_boxes_b, det_scores_b, det_labels_b,
    gt_boxes_b,  gt_labels_b,
    num_classes,
    iou_thr=0.5,
    score_thr=0.0
):

    db = [np.asarray(x, dtype=float) for x in det_boxes_b]
    ds = [np.asarray(x, dtype=float) for x in det_scores_b]
    dl = [np.asarray(x, dtype=int)   for x in det_labels_b]
    gb = [np.asarray(x, dtype=float) for x in gt_boxes_b]
    gl = [np.asarray(x, dtype=int)   for x in gt_labels_b]

    B = len(db)
    report = {c: {"TP":0,"FP":0,"FN":0,"precision":0.0,"recall":0.0,"f1":0.0,"support":0}
              for c in range(1, num_classes+1)}

    # build per-image GT pools
    gts = []
    for b in range(B):
        pool = {}
        for c in range(1, num_classes+1):
            mask = (gl[b] == c)
            g = gb[b][mask] if gb[b].size else gb[b]
            pool[c] = {"boxes": g, "matched": np.zeros(len(g), dtype=bool)}
        gts.append(pool)

    # accumulate detections per class
    for c in range(1, num_classes+1):
        dets = []
        for b in range(B):
            if len(dl[b]) == 0:
                continue
            m = (dl[b] == c)
            if score_thr > 0:
                m = m & (ds[b] >= score_thr)
            idxs = np.where(m)[0]
            for i in idxs:
                dets.append((float(ds[b][i]), b, db[b][i].astype(float)))
        dets.sort(key=lambda t: -t[0])

        tp=fp=0
        support = sum(len(gts[b][c]["boxes"]) for b in range(B))
        for score, b, box in dets:
            gt_boxes = gts[b][c]["boxes"]
            if len(gt_boxes) == 0:
                fp += 1; continue
            ious = iou_xyxy(box, gt_boxes)  # your IoU helper
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr and not gts[b][c]["matched"][j]:
                tp += 1; gts[b][c]["matched"][j] = True
            else:
                fp += 1

        fn = sum(int((~gts[b][c]["matched"]).sum()) for b in range(B))
        prec = tp/(tp+fp+1e-9); rec = tp/(support+1e-9)
        f1   = (2*prec*rec)/(prec+rec+1e-9)
        report[c] = {"TP":tp,"FP":fp,"FN":fn,
                     "precision":float(prec),"recall":float(rec),
                     "f1":float(f1),"support":int(support)}
    return report

def evaluation_pr_curve_single_class(
    c,
    det_boxes_b, det_scores_b, det_labels_b,
    gt_boxes_b,  gt_labels_b,
    iou_thr=0.5
):
    """
    Returns (rec, prec, scores_sorted) arrays for class c in 1..C.
    """
    db = np.asarray(det_boxes_b); ds = np.asarray(det_scores_b); dl = np.asarray(det_labels_b)
    gb = np.asarray(gt_boxes_b);  gl = np.asarray(gt_labels_b)
    B  = db.shape[0]

    # GT pool for class c
    gts = {b: {"boxes": gb[b][gl[b]==c].astype(float), "matched": np.zeros(np.sum(gl[b]==c), dtype=bool)}
           for b in range(B)}
    npos = int(sum(len(gts[b]["boxes"]) for b in range(B)))

    dets = []
    for b in range(B):
        mask = (dl[b]==c)
        for i in np.where(mask)[0]:
            dets.append((float(ds[b, i]), b, db[b, i].astype(float)))
    if len(dets)==0 or npos==0:
        return np.array([0.]), np.array([0.]), np.array([])

    dets.sort(key=lambda t: -t[0])
    tp = np.zeros(len(dets)); fp = np.zeros(len(dets)); scores = np.array([d[0] for d in dets])

    for i, (_, b, box) in enumerate(dets):
        gt_boxes = gts[b]["boxes"]
        if gt_boxes.shape[0] == 0:
            fp[i] = 1.0; continue
        ious = iou_xyxy(box, gt_boxes)
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr and not gts[b]["matched"][j]:
            tp[i] = 1.0; gts[b]["matched"][j] = True
        else:
            fp[i] = 1.0

    tp_c = np.cumsum(tp); fp_c = np.cumsum(fp)
    rec  = tp_c / (npos + 1e-9)
    prec = tp_c / (tp_c + fp_c + 1e-9)
    return rec, prec, scores


def evaluation_ece_from_logits(roi_labels, classification_head, n_bins=15):
    """
    Expected Calibration Error for RoI head.
    roi_labels: (B, N) with -1 ignored, 0=BG, 1..C-1=FG classes
    classification_head: (B, N, C) logits
    """
    labels = tf.convert_to_tensor(roi_labels)
    logits = tf.convert_to_tensor(classification_head)
    mask = labels != -1
    if not tf.reduce_any(mask):
        return tf.constant(0.0, tf.float32)

    y = tf.boolean_mask(labels, mask)          # (M,)
    p = tf.nn.softmax(tf.boolean_mask(logits, mask), axis=-1)  # (M, C)
    conf = tf.reduce_max(p, axis=-1)           # (M,)
    pred = tf.argmax(p, axis=-1, output_type=y.dtype)

    correct = tf.cast(tf.equal(pred, y), tf.float32)
    # binning
    bins = tf.linspace(0.0, 1.0, n_bins+1)
    ece  = tf.constant(0.0, tf.float32)
    m = tf.cast(tf.shape(conf)[0], tf.float32)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        sel = tf.where((conf > lo) & (conf <= hi))[:,0]
        if tf.size(sel) == 0: 
            continue
        conf_bin = tf.gather(conf, sel)
        corr_bin = tf.gather(correct, sel)
        gap = tf.abs(tf.reduce_mean(conf_bin) - tf.reduce_mean(corr_bin))
        ece += (tf.cast(tf.size(sel), tf.float32)/m) * gap
    return ece


def evaluation_map_by_area(
    det_boxes_b, det_scores_b, det_labels_b,
    gt_boxes_b,  gt_labels_b,
    num_classes,
    iou_thr=0.5,
    area_bins=((0, 32*32), (32*32, 96*96), (96*96, 10**9))
):
    """
    Computes mAP separately for GT size buckets. Returns dict: {'small':..., 'medium':..., 'large':...}
    """
    def area(boxes):
        w = np.maximum(0.0, boxes[:,2]-boxes[:,0])
        h = np.maximum(0.0, boxes[:,3]-boxes[:,1])
        return w*h

    names = ["small","medium","large"][:len(area_bins)]
    results = {}
    for (lo, hi), name in zip(area_bins, names):
        # mask GT by area; keep detections as-is
        gt_boxes_f, gt_labels_f = [], []
        det_boxes_f, det_scores_f, det_labels_f = [], [], []
        for b in range(len(gt_boxes_b)):
            a = area(np.asarray(gt_boxes_b[b]))
            gm = (a >= lo) & (a < hi)
            gt_boxes_f.append(np.asarray(gt_boxes_b[b])[gm])
            gt_labels_f.append(np.asarray(gt_labels_b[b])[gm])
            det_boxes_f.append(np.asarray(det_boxes_b[b]))
            det_scores_f.append(np.asarray(det_scores_b[b]))
            det_labels_f.append(np.asarray(det_labels_b[b]))
        mAP_bucket, _ = compute_map_voc(
            det_boxes_f, det_scores_f, det_labels_f,
            gt_boxes_f,  gt_labels_f,
            num_classes=num_classes, iou_thr=iou_thr, use_11pt=False
        )
        results[name] = float(mAP_bucket)
    return results


def evaluation_sanity_checks(
    det_boxes_b, det_scores_b, det_labels_b,
    gt_boxes_b,  gt_labels_b
):
    """
    Quick integrity checks; returns dict with booleans and counts.
    """
    db = np.asarray(det_boxes_b); ds = np.asarray(det_scores_b); dl = np.asarray(det_labels_b)
    gb = np.asarray(gt_boxes_b);  gl = np.asarray(gt_labels_b)

    def bad_boxes(x):
        x = x.reshape(-1, 4)
        return np.sum((x[:,2] <= x[:,0]) | (x[:,3] <= x[:,1]) | np.isnan(x).any(axis=1) | np.isinf(x).any(axis=1))

    return {
        "det_bad_boxes": int(bad_boxes(db)),
        "gt_bad_boxes" : int(bad_boxes(gb)),
        "det_nan_scores": int(np.isnan(ds).sum()),
        "det_inf_scores": int(np.isinf(ds).sum()),
        "det_neg_scores": int((ds < 0).sum()),
        "det_label_out_of_range": int(((dl < 0) | (dl != np.floor(dl))).sum()),
        "gt_label_out_of_range" : int(((gl < 0) | (gl != np.floor(gl))).sum()),
    }

def bias_init_for_prior(pi=0.01):
    return tf.keras.initializers.Constant(-math.log((1 - pi) / pi))