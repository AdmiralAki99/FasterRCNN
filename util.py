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

import matplotlib.colors as mcolors
import random

# Function to get the number of anchor points in the feature map

def get_number_of_anchor_points(feature_map) -> (int, int, int):
    """
    Calculates the number of anchor points (centers) for the the feature_map

    Parameters
    ----------
    feature_map: Feature Map created by CNN backbone
    ----------

    Returns
    ----------
    Tuple[int, int, int]
        Tuple of the Number of anchor points, the X-axis size of the feature map, the Y-axis size of the feature map
    ----------
    """

    if len(feature_map.shape) != 4:
        raise ValueError("Input must be a 4D tensor with shape (batch_size, height, width, channels)")
    
    # Get the Shape of the Image (WxH)
    _, axis_1, axis_2, _ = feature_map.shape
    # Total Number of Anchor Points Possible is WxH
    anchors = axis_1 * axis_2
    # Return Tuple of Number of Anchors, Axis Sizes
    return anchors, axis_1, axis_2

# Function to calculate the anchor stride to convert image space to feature map space

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

# Function to create anchor centers on the feature map

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


# Function to calculate the aspect ratio boxes for different scales and ratios

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

# Function to create anchor box relative to the center coordinates

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

# Function to initialize all the anchor boxes for the Faster R-CNN model

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

# Function to calculate the IoU scores for the ground truth boxes and all the anchor boxes

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

    # print(f'Coordinates Of Box 1: x1:{x11},y1:{y11},x2:{x21},y2:{y21}')
    # print(f'Shapes Of Box 1 Coordinates: x1:{x11.shape},y1:{y11.shape},x2:{x21.shape},y2:{y21.shape}')
    # print(f'Coordinates Of Box 2: x1:{x12},y1:{y12},x2:{x22},y2:{y22}')
    # print(f'Shapes Of Box 2 Coordinates: x1:{x12.shape},y1:{y12.shape},x2:{x22.shape},y2:{y22.shape}')

    x11 = tf.expand_dims(x11,axis=2)
    y11 = tf.expand_dims(y11,axis=2)
    x21 = tf.expand_dims(x21,axis=2)
    y21 = tf.expand_dims(y21,axis=2)

    x12 = tf.expand_dims(x12,axis=1)
    y12 = tf.expand_dims(y12,axis=1)
    x22 = tf.expand_dims(x22,axis=1)
    y22 = tf.expand_dims(y22,axis=1)

    # print(f'X11:{x11} Shape:{x11.shape}')
    # print(f'Y11:{y11} Shape:{y11.shape}')
    # print(f'X21:{x21} Shape:{x21.shape}')
    # print(f'Y21:{y21} Shape:{y21.shape}')
    
    # Now we to get the max and min values for each axes for each top left and bottom right corner of the box
    x1_max = tf.math.maximum(x11,x12)
    y1_max = tf.math.maximum(y11,y12)

    x2_min = tf.math.minimum(x21,x22)
    y2_min = tf.math.minimum(y21,y22)

    # print(f'X1 Max: {x1_max[0]}')
    # print(f'Y1 Max: {y1_max[0]}')
    
    # print(f'X2 Min: {x2_min}')
    # print(f'Y2 Min: {y2_min}')

    # print(f'Top Left Intersection Coordinates {tf.concat([x1_max[0],y1_max[0]],axis=1)}')
    # print(f'Bottom Right Intersection Coordinates {tf.concat([x2_min[0],y2_min[0]],axis=1)}')


    # Now we need to calculate the width and height of the intersection box, we need to stop it from being non-negative as well
    width = tf.math.maximum(0.0,x2_min - x1_max)
    height = tf.math.maximum(0.0,y2_min - y1_max)

    # print(f'Width Of Intersection: {width}')
    # print(f'Height Of Intersection: {height}')

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

    # print(f'Area of GT Boxes : {ground_truth_boxes_area}')
    # print(f'Area of Pred Boxes: {predicted_boxes_area}')
    # print(f'Area of Intersection: {area_of_intersection}')
    # print(f'Area of Union: {area_of_union}')
    # print(f'IOU Scores: {(iou_scores}')
    
    return tf.squeeze(iou_scores, axis=-1)

# Function to convert the coordinates to center format
# TODO: Remove this is redundant and reimplemented later

def convert_bounding_box_format(anchor_boxes):
    """
    Calculate the anchor box format from (x1,y1,x2,y2) to (xc,yc,w,h) so that the refinements can be calculated

    Parameters:
    ---------
    anchor_boxes : Tensor
        Tensor Of anchor boxes (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_OF_ANCHORS_PER_PIXEL,4)

    Returns:
    -------
    converted_boxes: Tensor
        Tensor Of Anchor boxes (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_OF_ANCHORS_PER_PIXEL,4)
        NOTE: Used for anchor box offset refinements only
    """
    # Split the coordinates into their individual points for both the axes
    x1,y1,x2,y2 = tf.split(anchor_boxes,num_or_size_splits=4,axis=-1)

    # There is an issue where splitting creates a new dimension in the tensor making the shape 
    # from (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_OF_ANCHORS) to (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_OF_ANCHORS,1)
    x1 = tf.squeeze(x1,axis=-1)
    x2 = tf.squeeze(x2,axis=-1)
    y1 = tf.squeeze(y1,axis=-1)
    y2 = tf.squeeze(y2,axis=-1)

    # Calculating the centers and the width and height of the boxes
    xc = (x2+x1) / 2.0
    yc = (y2+y1) / 2.0
    w = (x2 - x1)
    h = (y2 - y1)

    # Stacking the boxes after squeezing makes the shape to be (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_OF_ANCHORS,4)
    # instead of (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_OF_ANCHORS,1,4)
    converted_boxes = tf.stack([xc,yc,w,h],axis=-1)

    return converted_boxes

# Function to assign a positive or negative labels, for each ground truth box and anchor box combo

def assign_object_label(iou_scores_tensor,IOU_FOREGROUND_THRESH = 0.7,IOU_BACKGROUND_THRESH = 0.4,FEATURE_MAP_WIDTH= 50,FEATURE_MAP_HEIGHT= 50,NUM_OF_ANCHORS_PER_PIXEL= 9):
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
    object_label_tensor = tf.where(max_iou_per_anchor_box >IOU_FOREGROUND_THRESH,1,object_label_tensor)
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

# Function to generate the objectness labels (0-background,1-foreground,-1-Ignore) for the anchor boxes on the feature map

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

# Function to get the positive anchor boxes and their corresponding offsets

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
        Positive anchor boxes in the shape of (NUM_POS_ANCHORS,4) in the XY Coordinate system
        
    offsets_for_positive_anchor_boxes:
        Offsets for the positive anchor boxes in the shape(NUM_POS_ANCHORS , 4) in the Xc,Yc,W,H system
    """
    # Creating the positive mask and then getting the indices
    # We need to gather the bounding boxes
    positive_mask = tf.squeeze(object_labels,axis=-1) == 1 # Squeezing the last axis from the object labels
    positive_indices = tf.where(positive_mask) # Shape of (NUM_POS_ANCHOR_BOXES,4) [B, H, W, BOX_NO]

    # Need to calculate them per image in the batch not batchwise
    batch_indices = positive_indices[:,0] # Getting the batch index from the positive anchors
    batch_size = tf.reduce_max(batch_indices) + 1 # Zero-indexing for batches

    # Getting the positive anchors
    positive_anchors = []

    for index in range(batch_size):
        # Create a batch mask for the indices to select one batch at a time
        batch_mask = tf.equal(batch_indices,index)
        # Getting positive anchor boxes indices in the batch
        positive_anchor_boxes_in_batch = tf.boolean_mask(positive_indices,batch_mask)
        positive_anchors.append(positive_anchor_boxes_in_batch)

    # Padding the batches
    max_pos_anchors = max(tensor.shape[0] for tensor in positive_anchors)
    padded_anchors = []
    # Iterating over the batches and padding them
    for batch in positive_anchors:
        # Padding the anchors
        padding = [[0,max_pos_anchors-batch.shape[0]],[0,0]]
        padded_tensor = tf.pad(batch,padding,constant_values = -1)
        padded_anchors.append(padded_tensor)

    # Stacking the padded tensors together
    positive_anchors_indices = tf.stack(padded_anchors)

    # Using the same indices to get the offsets for the anchor_boxes
    offsets_for_positive_anchor_boxes = tf.gather_nd(offsets,positive_anchors_indices)

    # Getting the positive anchor boxes
    positive_anchors = tf.gather_nd(anchor_boxes,positive_anchors_indices)

    # return positive_anchor_boxes, offsets_for_positive_anchor_boxes,positive_indices
    return positive_anchors,offsets_for_positive_anchor_boxes,positive_anchors_indices

# Function to calculate objectness loss for the RPN

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
    retained_labels = tf.where(target_object_labels == -1, tf.zeros_like(target_object_labels), target_object_labels)
    binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()
    objectness_loss = binary_cross_entropy(retained_labels,predicted_scores)

    return objectness_loss

# Function to convert bounding boxes from (X_MIN,Y_MIN,X_MAX,Y_MAX) to (Xc,Yc,W,H)

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

# Function to convert bounding boxes from (Xc,Yc,W,H) to (X_MIN,Y_MIN,X_MAX,Y_MAX)

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

# Function to calculate the bounding box deltas between the RPN predicted anchor boxes and the anchor boxes

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

# Function to calculate the bounding box deltas between the anchor boxes and the ground truth boxes

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

    corresponding_gt_boxes = convert_bounding_box_format(corresponding_gt_boxes) # Convert the coordinate system from (X_MIN,Y_MIN,X_MAX,Y_MAX) to (Xc,Yc,W,H)
    positive_anchor_boxes = convert_bounding_box_format(positive_anchor_boxes)   # Convert the coordinate system from (X_MIN,Y_MIN,X_MAX,Y_MAX) to (Xc,Yc,W,H)

    # Splitting the anchor boxes into xc,yc,w,h separately
    gt_xc,gt_yc,gt_w,gt_h = tf.split(corresponding_gt_boxes,num_or_size_splits = 4,axis=-1)

    # Splitting the predicted boxes into xc,yc,w,h separately
    pred_xc,pred_yc,pred_w,pred_h = tf.split(positive_anchor_boxes,num_or_size_splits = 4,axis=-1)

    # According to the paper the deltas, the coefficients are calculated for the anchor box as well as
    # ground truth box called t & t*. These are only for positive anchors only
    tx = (pred_xc - gt_xc) / (gt_w + 1e-6)
    ty = (pred_yc - gt_yc) / (gt_h + 1e-6)
    tw = tf.math.log(tf.maximum((pred_w/(gt_w + 1e-6)),1e-6))
    th = tf.math.log(tf.maximum((pred_h/(gt_h + 1e-6)),1e-6))

    # Squeezing the last dimension
    tx = tf.squeeze(tx,axis=-1)
    ty = tf.squeeze(ty,axis=-1)
    tw = tf.squeeze(tw,axis=-1)
    th = tf.squeeze(th,axis=-1)
    
    t = tf.stack([tx,ty,tw,th],axis=-1)

    return t

# Function to calculate the Smooth L1 loss for offsets t and t*

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
    loss = tf.math.reduce_mean(tf.where(abs_loss < beta,((abs_loss ** 2)/2),((abs_loss - beta)/2))) # Mean since the formula uses MAE formula
    
    return loss

# Function to calculate the bounding box regression loss for the RPN

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
        Tensor of the ground truth boxes (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,NUM_ANCHORS,4)

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
    # First the bounding boxes need to be converted from [x1,y1,x2,y2] format to [xc,yc,w,h]
    formatted_anchor_boxes = convert_bounding_box_format(anchor_boxes * anchor_scaling_stride) # Converted anchor boxes into format [xc,yc,w,h]
    formatted_gt_boxes = convert_bounding_box_format(gt_boxes) # Converted ground truth boxes into format [xc,yc,w,h]

    # After conversion the offsets t and t* are calculated
    t_offset = calculate_bounding_box_deltas_between_pred_and_anchor_box(formatted_anchor_boxes,offsets)
    t_star_offset = calculate_bounding_box_deltas_between_gt_boxes(formatted_gt_boxes,formatted_anchor_boxes,iou_matrix,object_labels)

    # After calculating t and t* the positive anchors from t are retained and smooth l1 loss is calculated amongst them
    positive_mask = tf.squeeze(object_labels,axis=-1) == 1 # Squeezed the last dimension to make it more intuitive, created a binary mask for the positive anchors
    positive_offsets = tf.boolean_mask(t_offset,positive_mask) # Gathered all the positive anchors in the data

    # After gathering the offsets for the positive anchor boxes, the smooth l1 loss can be calulcated for it
    loss = smooth_l1_loss(positive_offsets,t_star_offset)

    return loss

# Function to calculate the loss of the RPN model using the objectness and bounding box regression loss

def calculate_rpn_loss(anchor_boxes,offsets,gt_boxes,iou_matrix,object_labels,objectness_scores,anchor_scaling_stride = 16,lambda_ = 10):
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

    lambda_: Float
        Balancing parameter based on the research paper to not allow one loss to dominate the other

    Returns:
    -------
    total_loss: Float
        Total loss value for the RPN model
    """ 
    # Calculate the number of positive_anchors
    num_positive_anchor_boxes = tf.reduce_sum(tf.cast(object_labels == 1, tf.float32))
    num_negative_anchor_boxes = tf.reduce_sum(tf.cast(object_labels == -1, tf.float32))

    # Calculate the normalizing terms
    n_cls = tf.maximum(num_positive_anchor_boxes + num_negative_anchor_boxes, 1e-6)
    n_reg = tf.maximum(num_positive_anchor_boxes, 1e-6)

    objectness_loss = calculate_objectness_loss(objectness_scores[...,1],object_labels)
    regression_loss = calculate_bounding_box_regression_loss(anchor_boxes,offsets,gt_boxes,iou_matrix,object_labels,anchor_scaling_stride = anchor_scaling_stride)
    # Calculate the total loss based on the paper
    total_loss =  ((1/n_cls) * (objectness_loss)) + (lambda_ * (1/n_reg) * regression_loss) 

    return total_loss

# Function to refine the region of interests (RoI) for the positive anchor boxes

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

    # Adding them anchor boxes with the offsets in the opposite way while calulating the differences in the offsets according to the paper.
    roi_xc = x_c + (t_xc * w)
    roi_yc = y_c + (t_yc * h)
    roi_w = tf.exp(t_w) * w
    roi_h = tf.exp(t_h) * h

    roi_proposals = tf.concat([roi_xc,roi_yc,roi_w,roi_h],axis=-1) # Concat them to create them to box coordinates

    return roi_proposals     # Created the RoI's, they need to be clipped when converting to the xy-coordinate system

# Function to convert the RoI to the xy-coordinate system

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

# Function to divide the RoI into equal sized grids

def divide_roi_into_grids(feature_map,bounding_boxes,box_indices,grid_row_size,grid_col_size):
    """
    Dividing the RoI's into grids and blocks that are pooled to be used in the RoI Head for the loss
    Parameters:
    ---------
    feature_map: Tensor
        Feature Map from the RPN (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,CHANNELS)

    bounding_boxes : Tensor
        Tensor of bounding boxes coordinates in the format (X_MIN,Y_MIN,X_MAX,Y_MAX) and the shape of (NUM_RoIs,4)

    box_indices : [Int]
        Indices of the batches for the bounding boxes

    grid_row_size: Int
        Row size for the cropped and resized grid

    grid_col_size: int
        Column size for the cropped and resized grid
        
    Returns:
    -------
    roi: Tesnor
        RoI's in the cropped and resized shape of (NUM_ROI,GRID_ROW_SIZE,GRID_COL_SIZE,CHANNELS)
    """
    
    feature_map_width = feature_map.shape[1]
    feature_map_height = feature_map.shape[2]

    # Splitting the bounding boxes
    x_min,y_min,x_max,y_max = tf.split(bounding_boxes, num_or_size_splits = 4, axis=-1)

    # print(f'X_MIN: {x_min}, Y_MIN: {y_min}, X_MAX: {x_max}, Y_MAX: {y_max}')

    # Tensorflow crop and resize requires the 'boxes' to be normalized between [0,1]
    x_min_normalized = x_min / tf.cast(feature_map_width,tf.float32)
    y_min_normalized = y_min / tf.cast(feature_map_height,tf.float32)
    x_max_normalized = x_max / tf.cast(feature_map_width,tf.float32)
    y_max_normalized = y_max / tf.cast(feature_map_height,tf.float32)

    # print(f'X_MIN_NORM: {x_min_normalized}, Y_MIN_NORM: {y_min_normalized}, X_MAX_NORM: {x_max_normalized}, Y_MAX_NORM: {y_max_normalized}')

    # # Stacking the normalized tensor to be used in the image.crop_and_resize
    normalized_bounding_boxes = tf.concat([y_min_normalized, x_min_normalized, y_max_normalized, x_max_normalized],axis=-1)
    valid_mask = tf.reduce_any(tf.not_equal(normalized_bounding_boxes,0.0),axis=-1) # Masking for the padded functions

    normalized_bounding_boxes = tf.boolean_mask(normalized_bounding_boxes,valid_mask) # Gathering and Stacking them together
  

    # Need to make sure that the function receives the coordinates in the shape (Y_MIN,X_MIN,Y_MAX,X_MAX)
    roi = tf.image.crop_and_resize(feature_map,normalized_bounding_boxes,tf.cast(box_indices,tf.int32),(grid_row_size,grid_col_size))

    return roi

# Function for RoI pooling, that will take the positive anchor boxes and making it into a fixed size to be flattened and used in the RoI head

def roi_pooling(feature_map,positive_anchor_boxes,positive_indices,offsets_for_anchor_boxes,output_size_x = 7,output_size_y = 7):
    """
    RoI pooling for the positive anchor boxes/regions generated by the RPN
    Parameters:
    ---------
    feature_map: Tensor
        Feature Map from the RPN (B,FEATURE_MAP_WIDTH,FEATURE_MAP_HEIGHT,CHANNELS)

    positive_anchor_boxes : Tensor
        Tensor of bounding boxes coordinates in the format (X_MIN,Y_MIN,X_MAX,Y_MAX) and the shape of (NUM_POS_ANCHOR_BOXES,4)

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
        RoI's in the cropped and resized shape of (NUM_ROI,GRID_ROW_SIZE,GRID_COL_SIZE,CHANNELS)
    """

    # Convert the bounding boxes from (X_MIN,Y_MIN,X_MAX,Y_MAX) to (Xc,Yc,W,H)
    converted_positive_anchor_boxes = convert_xy_boxes_to_center_format(positive_anchor_boxes)
    
    # Need to create the RoI's before pooling them in the correct size
    region_of_interests = refine_region_of_interests(converted_positive_anchor_boxes,offsets_for_anchor_boxes)
    bounding_boxes = convert_center_format_boxes_to_xy_coordinate(region_of_interests)
   
    # Filtering the valid indices after padding done before
    valid_mask = tf.reduce_any(tf.not_equal(positive_indices,-1),axis=-1)
    positive_indices = tf.boolean_mask(positive_indices,valid_mask)

    # Divide RoI's into Grids
    blocks = divide_roi_into_grids(feature_map,bounding_boxes,positive_indices[:,0],output_size_x,output_size_y)

    return blocks,bounding_boxes

# Function to assign RoI to Ground Truth Boxes

def assign_roi_to_ground_truth_box(ground_truth_boxes,roi_coordinates,ground_truth_labels,image_stride = 16):
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
    
    # Calculate the roi coordinates back in to image space
    image_roi_coordinates = roi_coordinates * image_stride
    
    # Calculate the IoU score
    iou_score = IOU_scores(ground_truth_boxes,image_roi_coordinates)

    # Get the best anchor box for each ground truth box
    best_roi_for_each_gt_boxes = tf.argmax(iou_score,axis = -1)

    # Getting the best ground truth for anchor boxes
    best_gt_box_for_each_anchor_box = tf.argmax(iou_score,axis = 1)

    # Getting Max IOU for Each RoI
    max_iou_per_anchor_box = tf.reduce_max(iou_score, axis=1)

    # Check if IOU value is more than the threshold
    roi_labels = tf.gather(ground_truth_labels,best_gt_box_for_each_anchor_box,batch_dims=1)

    # Applying IoU thresholding and making it a background if less than threshold
    roi_labels = tf.where(max_iou_per_anchor_box >= 0.5, roi_labels, tf.zeros_like(roi_labels))
    
    return roi_labels, best_gt_box_for_each_anchor_box, max_iou_per_anchor_box

# Function to match ground truth boxes to the RoI coordinates

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

# Function to calculate the bounding box deltas between the RoI and the GT boxes

def calculate_bounding_box_deltas_between_roi_and_ground_truth_box(ground_truth_boxes,roi_coordinates, roi_labels,image_space_stride = 16):
    """
    Calculate bounding box deltas between RoI coordinate boxes and ground truth boxes
    Parameters:
    ---------
    ground_truth_boxes: Tensor
        Ground truth box tensor (B,NUM_OF_GT_BOXES,4)

    roi_coordinates : Tensor
        Tensor of RoI coordinates in the format (X_MIN,Y_MIN,X_MAX,Y_MAX) and the shape of (B,NUM_OF_ROIS,4)
        
    Returns:
    -------
    filtered_t_offsets: Tensor
        Tensor of labels for each RoI in the shape of (B,NUM_OF_GT_BOXES,4)
    """
    # Converting the RoI coordinate from xy-format to the center format
    center_roi_coordinates = convert_xy_boxes_to_center_format(roi_coordinates * image_space_stride)
    # Converting the ground truth boxes from xy-format to center format
    center_ground_truth_boxes = convert_xy_boxes_to_center_format(ground_truth_boxes)

    # Calculate the offsets using the formula from the paper
    gt_center_x, gt_center_y, gt_w, gt_h = tf.split(center_ground_truth_boxes, num_or_size_splits = 4, axis = -1) # Splitting ground truth boxes
    roi_center_x, roi_center_y, roi_w, roi_h = tf.split(center_roi_coordinates, num_or_size_splits = 4, axis = -1) # Splitting the RoI's

    # Calculating the offsets
    t_x = (gt_center_x - roi_center_x)/(roi_w + 1e-6)
    t_y = (gt_center_y - roi_center_y)/(roi_h + 1e-6)
    t_w = tf.math.log(gt_w/(roi_w + 1e-6))
    t_h = tf.math.log(gt_h/(roi_h + 1e-6))
    
    t = tf.concat([t_x,t_y,t_w,t_h],axis=-1)

    # Calculating a valid mask based on the rois
    valid_mask = tf.reduce_any(roi_coordinates != 0.0, axis=-1)
    valid_mask = tf.reshape(valid_mask,[-1])

    # Filtering the t offsets using the same valid mask
    t = tf.reshape(t,[-1,4])
    filtered_t_offsets = tf.boolean_mask(t,valid_mask)

    # Filtering the RoI labels for Categorical Crossentropy in the RoI Head
    flattened_roi_labels = tf.reshape(roi_labels,[-1])
    # Getting the valid labels from them by using the same mask
    valid_roi_labels = tf.boolean_mask(flattened_roi_labels,valid_mask)
    
    return filtered_t_offsets,valid_roi_labels

"""========================================================================================================================================="""

# UTILITY: Function to display the center grid on the image

def display_center_grid(img, anchor_center_coord, number_of_anchors) -> np.array:
    """
    Creates a grid of the anchor points for the image

    Parameters
    ----------
    img: Input Image
    anchor_center_coord: Array of anchor centers
    number_of_anchors: Number of anchors for the Image
    ----------

    Returns
    ----------
    img_copy
        Image copy that has the bounding box created over it
    ----------
    """

    # Copy of the input image to be manipulated
    img_copy = np.copy(img)

    # Iterate over the image and create the anchor coordinate grid
    for i in range(number_of_anchors):
        cv.circle(img_copy, (int(anchor_center_coord[i][0]), int(anchor_center_coord[i][1])), radius=2,
                  color=(255, 0, 0), thickness=1)

    # img_copy = cv.addWeighted(img_copy,0.4,img,1-0.4,0)

    # Image with Anchor Grid
    return img_copy

# UTILITY: Function to create the anchor boxes on the image

def create_anchor_boxes(x1, y1, x2, y2, img) -> np.array:
    """
    Creates a grid of the anchor points for the image

    Parameters
    ----------
    x1: X-coordinate of Top Left Corner
    y1: Y-coordinate of Top Left Corner
    x2: X-coordinate of Bottom Right Corner
    y2: Y-coordinate of Bottom Right Corner
    colour: Color for the anchor boxes
    ----------

    Returns
    ----------
    img_box
        Image with anchor box
    ----------
    """
    # Creating Center Coordinate To Isolate Point For Better Understanding
    center_x = (x1+x2) // 2
    center_y = (y1+y2) // 2
    
    # Creating the bounding box centered over a pixel
    img_box = cv.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    img_box = cv.circle(img_box,(center_x,center_y),5,(0,255,0),-1)
    # Image with bounding box
    return img_box

# UTILITY: Function to display grid of all the relative anchor boxes at one center point

def create_relative_anchor_boxes(anchor_boxes, img):
    """
    Creates a grid of the anchor points for the image

    Parameters
    ----------
    image_tensor: Tensor with the image Anchor box
    img: Input Image
    colour: Color of the bounding box
    ----------

    Returns
    ----------
    None
    ----------
    """
    """
    
    """
    for coordinates in anchor_boxes:
            x1 = int(coordinates[0])
            y1 = int(coordinates[1])
            x2 = int(coordinates[2])
            y2 = int(coordinates[3])
            img = create_anchor_boxes(x1, y1, x2, y2, img)
        # Image with multiple bounding boxes over a central pixel
    return img

# UTILITY: Utility Function To Generate Ground Truth Boxes For Testing IOU
def generate_sample_ground_truth_boxes(image_width = 800,image_height = 800,num_gt_boxes = 3):
     # Generate coordinates for the top-left corner
    x1 = tf.random.uniform(shape=(num_gt_boxes,), minval=0, maxval=image_width // 2, dtype=tf.float32)
    y1 = tf.random.uniform(shape=(num_gt_boxes,), minval=0, maxval=image_height // 2, dtype=tf.float32)

    # Generate coordinates for the bottom-right corner
    x2 = tf.clip_by_value(x1 + tf.random.uniform(shape=(num_gt_boxes,), minval=50, maxval=200, dtype=tf.float32), 0, image_width)
    y2 = tf.clip_by_value(y1 + tf.random.uniform(shape=(num_gt_boxes,), minval=50, maxval=200, dtype=tf.float32), 0, image_height)

    # Stack and add batch dimension
    ground_truth_boxes = tf.stack([x1, y1, x2, y2], axis=1)
    return tf.expand_dims(ground_truth_boxes, axis=0)  # Shape (B, NO_OF_GT_BOXES, 4)


# UTILITY: Function To Generate Ground Truth Labels For Classification Score Testing In RoI Head

def generate_sample_ground_truth_labels(num_gt_boxes = 3, min_label_value = 0, max_label_value = 10):
    # Generating the labels for the ground truth boxes
    labels = tf.random.uniform(shape=(num_gt_boxes,),minval=min_label_value,maxval=max_label_value, dtype=tf.int32)

    return labels

# UTILITY: Function To Generate Sample Anchor Boxes For IoU Testing

def generate_sample_anchor_boxes(image_width = 800, image_height = 800, num_anchor_boxes = 5):
    # Predicted Boxes (top-left (x1, y1), bottom-right (x2, y2))
    # Generating coordinates for the top-left corner
    x1 = tf.random.uniform(shape=(num_anchor_boxes,), minval=0, maxval=image_width // 2, dtype=tf.float32)
    y1 = tf.random.uniform(shape=(num_anchor_boxes,), minval=0, maxval=image_width // 2, dtype=tf.float32)

    # Generating coordinates for the bottom-right corner
    x2 = x1 + tf.random.uniform(shape=(num_anchor_boxes,), minval=50, maxval=200, dtype=tf.float32)
    y2 = y1 + tf.random.uniform(shape=(num_anchor_boxes,), minval=50, maxval=200, dtype=tf.float32)

    # Stacking the coordinates along axis 1 to get shape (num_gt_boxes, 4)
    anchor_boxes = tf.stack([x1, y1, x2, y2], axis=1)
    
    return tf.expand_dims(anchor_boxes,axis=0) # Creating Tensor of Shape (B,NO_OF_PRED_BOXES,BOX_COORDINATES)

# UTILITY: Function To Check IOU intersection
def plot_bounding_boxes(ground_truth_boxes, predicted_boxes, size=100):
    # Ensure size is within the desired range
    size = min(max(size, 100), 800)  # Constrain to between 100 and 800
    figure_scale = size / 100  
    fig_width, fig_height = 8 * figure_scale, 8 * figure_scale
    
    for batch in range(ground_truth_boxes.shape[0]):
        # Split ground truth boxes into four coordinates
        gt_x1, gt_y1, gt_x2, gt_y2 = tf.split(ground_truth_boxes[batch, :, :], num_or_size_splits=4, axis=-1)
        
        # Split predicted boxes into four coordinates
        pred_x1, pred_y1, pred_x2, pred_y2 = tf.split(predicted_boxes[batch, :, :], num_or_size_splits=4, axis=-1)
        
        iou_scores = IOU_scores(ground_truth_boxes, predicted_boxes)[batch, :, :]
        
        fig, axs = plt.subplots(ground_truth_boxes.shape[1], figsize=(fig_width, fig_height))
        
        for box_index in range(ground_truth_boxes.shape[1]):
            gt_bottom_left_x = gt_x1[box_index].numpy()
            gt_bottom_left_y = gt_y2[box_index].numpy()
            gt_width = gt_x2[box_index].numpy() - gt_x1[box_index].numpy()
            gt_height = gt_y1[box_index].numpy() - gt_y2[box_index].numpy()
            
            axs[box_index].add_patch(plt.Rectangle(
                (gt_bottom_left_x, gt_bottom_left_y),
                gt_width,
                gt_height,
                edgecolor='green',
                facecolor='none',
                linewidth=2,
                label='Ground Truth'
            ))
            
            axs[box_index].set_xlim([0, size])
            axs[box_index].set_ylim([0, size])
            axs[box_index].invert_yaxis()  # Optional: to match standard image coordinates
            axs[box_index].set_title(f'Ground Truth Box {box_index + 1}')
            axs[box_index].set_aspect('equal', adjustable='box')
            
            for pred_index in range(predicted_boxes.shape[1]):
                pred_bottom_left_x = pred_x1[pred_index].numpy()
                pred_bottom_left_y = pred_y2[pred_index].numpy()
                pred_width = pred_x2[pred_index].numpy() - pred_x1[pred_index].numpy()
                pred_height = pred_y1[pred_index].numpy() - pred_y2[pred_index].numpy()
                
                axs[box_index].add_patch(plt.Rectangle(
                    (pred_bottom_left_x, pred_bottom_left_y),
                    pred_width,
                    pred_height,
                    edgecolor='orange',
                    facecolor='none',
                    linewidth=1,
                    linestyle='--',
                    label='Predicted'
                ))
                
                axs[box_index].text(pred_x1[pred_index].numpy(), pred_y1[pred_index].numpy(),
                                    f'IOU: {iou_scores[box_index, pred_index]:.2f}',
                                    verticalalignment='bottom')
                
            axs[box_index].legend(['Ground Truth', 'Predicted'], loc='upper right')
        
        # plt.tight_layout()
        plt.show()
        
# UTILITY: Utility function to display anchor box around a center coordinate

def display_anchor_box_around_center_coordinate(anchor_boxes,center_x,center_y,plot_width = 50,plot_height = 50):
    batch_size,num_of_anchors,_ = anchor_boxes.shape
    colors = ['red','chartreuse','turquoise','deepskyblue','midnightblue','darkslateblue','darkviolet','deeppink','darkorange']
    for batch_index in range(batch_size):
        fig, axs = plt.subplots(anchor_boxes.shape[0], figsize=(plot_width, plot_height)) # We create spearate image for each batch and overlay all the anchor boxes on top of each other
        for box_index in range(anchor_boxes.shape[1]):
            # Iterate through the anchor boxes and then plot them
            anchor_box = anchor_boxes[batch_index,box_index,:]
            x1,y1,x2,y2 = tf.split(anchor_box,num_or_size_splits=4,axis=-1)
            width = x2 - x1
            height = y2 - y1
            
            axs.add_patch(plt.Rectangle(
                (x1, y1),
                width,
                height,
                edgecolor=colors[box_index],
                facecolor='none',
                linewidth=2,
                linestyle='--',
                label='Ground Truth'
            ))

            axs.plot(center_x, center_y, 'ro', label='Center Point')
            axs.plot(x1,y1,'bo',label="LB")
            axs.plot(x1 + width,y1 + height,'yo',label="RB")
            axs.set_xlim([0, plot_width])
            axs.set_ylim([0, plot_height])
            axs.invert_yaxis()
            axs.set_title(f'Ground Truth Box {box_index + 1}')
            axs.set_aspect('equal', adjustable='box')