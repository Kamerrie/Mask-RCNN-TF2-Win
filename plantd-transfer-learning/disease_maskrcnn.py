import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from numpy import zeros, asarray

import mrcnn
import mrcnn.utils
import mrcnn.config
import mrcnn.model
import tensorflow as tf

from datetime import datetime

# Dirs
DATASET_DIR = os.path.join(os.getcwd(), "data") # volume where the images wind up
MASKRCNN_DIR = os.path.join(os.getcwd(), "maskrcnn") # volume where we will keep logs/models(weights)
MODEL_DIR = os.path.join(MASKRCNN_DIR, "models") # the model directory within the mrcnn volume
#PRE_TRAINED_PATH

# Useful vars
total_train_images = len([name for name in os.listdir(os.path.join(DATASET_DIR, "train", "images")) if os.path.isfile(os.path.join(os.path.join(DATASET_DIR, "train", "images"), name)) and name.endswith('.jpg')])
total_val_images = len([name for name in os.listdir(os.path.join(DATASET_DIR, "val", "images")) if os.path.isfile(os.path.join(os.path.join(DATASET_DIR, "val", "images"), name)) and name.endswith('.jpg')])


class DiseaseDataset(mrcnn.utils.Dataset):

    def load_dataset(self, dataset_dir, subset):
        """Load a subset of the Plant Disease dataset.
        
        dataset_dir: The root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("dataset", 1, "plant_disease")

        # Define data locations
        subset_dir = os.path.join(dataset_dir, subset)
        image_dir = os.path.join(subset_dir, "images")
        mask_dir = os.path.join(subset_dir, "masks")

        # Load the images
        for image_id, image_name in enumerate(os.listdir(image_dir)):
            if not image_name.endswith(".jpg"):
                continue
            
            # Get image path and mask path
            image_path = os.path.join(image_dir, image_name)
            mask_path = os.path.join(mask_dir, image_name.replace('.jpg', '.png'))

            # Add image to the dataset
            self.add_image(
                "dataset",
                image_id=image_id,
                path=image_path,
                mask_path=mask_path
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        
        Returns:
        masks: A bool array of shape [height, width, instance count] with a binary mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        mask_path = info['mask_path']

        # Load the mask from disk
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)  # Convert to binary mask
        mask = np.expand_dims(mask, axis=-1)  # Add an extra dimension

        # Create an array of class IDs, since we only have one class "plant_disease"
        class_ids = np.array([1], dtype=np.int32)

        return mask, class_ids

class TrainConfig(mrcnn.config.Config):
    NAME = "train_cfg"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 2
    BACKBONE = "resnet50"
    STEPS_PER_EPOCH = total_train_images

training_config = TrainConfig()

class InferConfig(TrainConfig):
    NAME = "infer_cfg"
    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = total_val_images

inference_config = InferConfig()

# Training dataset
dataset_train = DiseaseDataset()
dataset_train.load_dataset(DATASET_DIR, "train")
dataset_train.prepare()

# Validation dataset
dataset_val = DiseaseDataset()
dataset_val.load_dataset(DATASET_DIR, "val")
dataset_val.prepare()

# Create model in training mode
model = mrcnn.model.MaskRCNN(mode="training", config=training_config, model_dir=MODEL_DIR)

# Load pre-trained weights
#model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=[
#    "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

# Setting up custom callbacks to collect logs, metrics, make checkpoints, and save the best model based on mean IoU
class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_file='maskrcnn.log'):
        super(MetricsCallback, self).__init__()
        self.log_file = log_file
        self.start_time = None
        self.best_mean_iou = 0.0  # Initialize best IoU to a low value
        self.best_checkpoint_path = 'best_model.h5'  # Path to save the best model

        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("start_time,epoch,end_time,epoch_duration,mean_iou,mean_precision,mean_recall,mean_f1_score\n")

    def on_train_begin(self, logs=None):
        self.start_time = datetime.now()
        formatted_start_time = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Training started at: {formatted_start_time}")

        with open(self.log_file, 'a') as f:
            f.write(f"{formatted_start_time},N/A,N/A,N/A,N/A,N/A,N/A,N/A\n")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f'\nEpoch {epoch + 1} Metrics:')
        end_time = datetime.now()
        epoch_duration = (end_time - self.start_time).total_seconds()
        formatted_end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")

        model_path = f'Disease_mask_rcnn_epoch_{epoch + 1}.h5'
        self.model.save_weights(model_path)

        # Create inference model after saving weights
        inference_model = mrcnn.model.MaskRCNN(mode='inference', 
                                               model_dir='./', 
                                               config=inference_config)
        inference_model.load_weights(model_path, by_name=True)

        # Initialize lists to store metric results for this epoch
        val_iou = []
        precisions, recalls, f1_scores = [], [], []
        
        for image_id in dataset_val.image_ids:
            image = dataset_val.load_image(image_id)
            mask, _ = dataset_val.load_mask(image_id)
            results = inference_model.detect([image], verbose=0)
            pred_mask = results[0]['masks']
            
            if pred_mask.shape[-1] > 0 and mask.shape[-1] > 0:
                # Compute IoU for each predicted mask with each true mask
                iou_matrix = np.zeros((pred_mask.shape[-1], mask.shape[-1]))
                for i in range(pred_mask.shape[-1]):
                    for j in range(mask.shape[-1]):
                        intersection = np.logical_and(pred_mask[:, :, i], mask[:, :, j])
                        union = np.logical_or(pred_mask[:, :, i], mask[:, :, j])
                        iou = np.sum(intersection) / np.sum(union)
                        iou_matrix[i, j] = iou
                
                # Match predicted masks to true masks based on IoU
                matches = np.argmax(iou_matrix, axis=1)
                
                for i, match in enumerate(matches):
                    max_iou = iou_matrix[i, match]
                    val_iou.append(max_iou)
                    
                    pred = pred_mask[:, :, i]
                    true = mask[:, :, match]
                    intersection = np.logical_and(pred, true)
                    union = np.logical_or(pred, true)
                    
                    precision = np.sum(intersection) / np.sum(pred) if np.sum(pred) > 0 else 0
                    recall = np.sum(intersection) / np.sum(true) if np.sum(true) > 0 else 0
                    precisions.append(precision)
                    recalls.append(recall)
                    
                    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
                    f1_scores.append(f1)
        
        mean_iou_value = np.mean(val_iou)
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)
        mean_f1_score = np.mean(f1_scores)

        logs['mean_iou'] = mean_iou_value
        logs['mean_precision'] = mean_precision
        logs['mean_recall'] = mean_recall
        logs['mean_f1_score'] = mean_f1_score

        if mean_iou_value > self.best_mean_iou:
            print(f"Mean IoU improved from {self.best_mean_iou:.4f} to {mean_iou_value:.4f}. Saving model checkpoint.")
            self.best_mean_iou = mean_iou_value
            self.model.save_weights(self.best_checkpoint_path)

        with open(self.log_file, 'a') as f:
            f.write(f"{self.start_time.strftime('%Y-%m-%d %H:%M:%S')},{epoch + 1},{formatted_end_time},{epoch_duration:.2f},{mean_iou_value:.4f},{mean_precision:.4f},{mean_recall:.4f},{mean_f1_score:.4f}\n")
        
        self.start_time = datetime.now()

        for metric_name, metric_value in logs.items():
            print(f'{metric_name}: {metric_value:.4f}')

# Training logic with Mask R-CNN setup
metrics_callback = MetricsCallback()


# Train the model
model.train(dataset_train, dataset_val,
            learning_rate=training_config.LEARNING_RATE,
            epochs=30,
            layers='heads',
            custom_callbacks=[metrics_callback])

