import os
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xml.etree.ElementTree as ET
from tensorflow_datasets.core import GeneratorBasedBuilder, DatasetInfo
from tensorflow_datasets import features


## Citation for the PASCAL VOC 2012 dataset from tensorflow_datasets
_CITATION = """\
@misc{pascal-voc-2012,
  title = {PASCAL Visual Object Classes Challenge 2012},
  url = {http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html}
}
"""

# labels for the PASCAL VOC 2012 dataset
## TODO: Check if the labels are correct
_LABELS = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

## Custom Loader For PASCAL VOC 2012

## Genrator Builder for the PASCAL VOC 2012 dataset based on the tensorflow_datasets library
## Needed to create this because of redirect error when using the default loader
class CustomVOC(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Download VOCtrainval_11-May-2012.tar from:

        http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit

    And place it in the manual_dir, typically:
        ~/tensorflow_datasets/downloads/manual/
    """

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="Custom Pascal VOC 2012 (train/val only)",
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(),
                "objects": tfds.features.Sequence({
                    "bbox": tfds.features.BBoxFeature(),
                    "label": tfds.features.ClassLabel(names=_LABELS),
                    "is_difficult": tfds.features.Tensor(shape=(), dtype=tf.bool),
                }),
            }),
            supervised_keys=None,
            homepage="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        extracted_path = dl_manager.extract(
            dl_manager.manual_dir / "VOCtrainval_11-May-2012.tar"
        )
        voc_path = os.path.join(extracted_path, "VOCdevkit", "VOC2012")
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={"voc_path": voc_path, "split": "train"},
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={"voc_path": voc_path, "split": "val"},
            ),
        ]

    def _generate_examples(self, voc_path, split):
        images_dir = os.path.join(voc_path, "JPEGImages")
        annotations_dir = os.path.join(voc_path, "Annotations")
        split_file = os.path.join(voc_path, "ImageSets", "Main", f"{split}.txt")

        with open(split_file) as f:
            image_ids = [line.strip() for line in f]

        for image_id in image_ids:
            annotation_path = os.path.join(annotations_dir, f"{image_id}.xml")
            tree = ET.parse(annotation_path)
            root = tree.getroot()

            objs = []
            for obj in root.findall("object"):
                label = obj.find("name").text
                difficult = int(obj.find("difficult").text or 0)
                bndbox = obj.find("bndbox")
                bbox = {
                    "ymin": float(bndbox.find("ymin").text) / int(root.find("size/height").text),
                    "xmin": float(bndbox.find("xmin").text) / int(root.find("size/width").text),
                    "ymax": float(bndbox.find("ymax").text) / int(root.find("size/height").text),
                    "xmax": float(bndbox.find("xmax").text) / int(root.find("size/width").text),
                }
                objs.append({
                    "bbox": tfds.features.BBox(**bbox),
                    "label": label,
                    "is_difficult": bool(difficult),
                })

            yield image_id, {
                "image": os.path.join(images_dir, f"{image_id}.jpg"),
                "objects": objs,
            }
            

# Function to preprocess the data into desired format
@tf.function
def preprocess_dataset(data: dict,target_shape=(800,800)):
    """
    Preprocessing the dataset from the Tensorflow dataset format to the format that needs to be given to the Faster RCNN model

    Parameters:
    -----------
    data: dict
        Dictionary of the data from the PASCAL VOC dataset

    target_shape: Tuple
        Tuple of the target size to resize the images for the model
    
    Returns:
    -------
    formatted_data: dict
        Dictionary of the dataset information for the next steps in the preprocessing pipeline
    """
    
    # Getting the information from the image
    image = data['image']
    ground_truth_boxes = data['objects']['bbox']
    ground_truth_labels = data['objects']['label']

    # Converting the image to tf.float32 for higher detail of data
    image = tf.image.convert_image_dtype(image,tf.float32)


    # Converting the bounding boxes to [X_MIN,Y_MIN,X_MAX,Y_MAX] from [Y_MIN,X_MIN,Y_MAX,X_MAX]
    # Removing the normalization from the boxes
    H = tf.cast(tf.shape(image)[0],tf.float32)
    W = tf.cast(tf.shape(image)[1],tf.float32)
    
    # Multiplying the Y-axis with the height, and the X-axis with the width
    ground_truth_boxes = tf.stack([ground_truth_boxes[:,1] * W,ground_truth_boxes[:,0] * H,ground_truth_boxes[:,-1] * W,ground_truth_boxes[:,2] * H],axis=1)

    # Resizing the image
    image = tf.image.resize(image,target_shape)

    # Scaling the ground truth boxes
    H_SCALE = tf.cast(target_shape[0],dtype=tf.float32) / H
    W_SCALE = tf.cast(target_shape[1],dtype = tf.float32) / W  
    
    ground_truth_boxes = tf.stack([ground_truth_boxes[:,0] * W_SCALE, ground_truth_boxes[:,1] * H_SCALE, ground_truth_boxes[:,2] * W_SCALE, ground_truth_boxes[:,-1] * H_SCALE],axis=1)

    # Returning the information in similar format to the original dataset.
    return image, ground_truth_boxes, ground_truth_labels
    
    
@tf.function
def map_dataset(data):
    """
    Preprocesses a single TFDS example for Faster R-CNN training.

    Parameters:
    ----------
    dict: Dict
         A dictionary containing a single example from the TFDS Pascal VOC dataset.

    Returns:
    ---------
    dict: Dict
        A dictionary with the following keys:
            - 'image': float32 image tensor of shape [H, W, 3], resized and normalized
            - 'gt_boxes': float32 tensor of shape [N, 4], absolute pixel bounding boxes 
                          in [X_MIN, Y_MIN, X_MAX, Y_MAX] format
            - 'gt_labels': int32 tensor of shape [N], class labels
        
    """
    # Calling the preprocessing function
    image, bboxes, labels = preprocess_dataset(data)
    
    return{
    'image': image,
    'gt_boxes': bboxes,
    'gt_labels': labels
    }

# Define the stacking function so that the batches can be created
@tf.function
def stack_batches(batch):
    images = tf.stack(batch['image'],axis=0)
    gt_boxes = [item for item in batch['gt_boxes']]
    gt_labels = [item for item in batch['gt_labels']]

    return {
        'images': images,
        'gt_boxes': gt_boxes,
        'gt_labels': gt_labels
    }

@tf.function
def prepare_faster_rcnn_dataset(dataset,batch_size = 2,target_size = (800,800)):
    """
    Function to prepare the dataset for the Faster RCNN model.
    
    Parameters:
    -----------
    dataset : tf.data.Dataset
        TensorFlow dataset loaded from the Pascal VOC dataset using TFDS.

    batch_size : int, optional
        Number of samples per batch (default is 2).

    target_size : tuple, optional
        Tuple specifying the target size (height, width) to which all images 
        should be resized (default is (800, 800)).

    Returns:
    --------
    tf.data.Dataset
        A batched and preprocessed dataset where each batch contains:
        - 'image': tensor of shape [B, H, W, 3], float32
        - 'gt_boxes': tensor of shape [B, max_N, 4], float32 bounding boxes
        - 'gt_labels': tensor of shape [B, max_N], int32 class labels
    """
    # Mapping the dataset to the preprocess functions
    dataset = dataset.map(map_dataset,num_parallel_calls=tf.data.AUTOTUNE)
    # Creating the padded batches for the dataset
    dataset = dataset.padded_batch(batch_size,drop_remainder=True)

    return dataset