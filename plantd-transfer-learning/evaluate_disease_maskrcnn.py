import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import tensorflow as tf

import mrcnn.model as modellib
import mrcnn.utils as utils
from mrcnn.config import Config

# Set the directory paths
DATASET_DIR = os.path.join(os.getcwd(), "data")
MASKRCNN_DIR = os.path.join(os.getcwd(), "maskrcnn")
MODEL_DIR = os.path.join(MASKRCNN_DIR, "models")
RESULT_DIR = os.path.join(MASKRCNN_DIR, "results")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "weights", "best_model.h5")

# Ensure RESULT_DIR exists
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

# Custom Dataset for Evaluation
class DiseaseDataset(utils.Dataset):
    def load_dataset(self, dataset_dir, subset):
        """Load a subset of the Plant Disease dataset."""
        self.add_class("dataset", 1, "plant_disease")
        
        subset_dir = os.path.join(dataset_dir, subset)
        image_dir = os.path.join(subset_dir, "images")
        mask_dir = os.path.join(subset_dir, "masks")

        for image_id, image_name in enumerate(os.listdir(image_dir)):
            if not image_name.endswith(".jpg"):
                continue
            image_path = os.path.join(image_dir, image_name)
            mask_path = os.path.join(mask_dir, image_name.replace('.jpg', '.png'))
            self.add_image("dataset", image_id=image_id, path=image_path, mask_path=mask_path)

    def load_mask(self, image_id):
        """Generate instance masks for an image."""
        info = self.image_info[image_id]
        mask_path = info['mask_path']
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)  # Convert to binary mask
        mask = np.expand_dims(mask, axis=-1)  # Add an extra dimension
        class_ids = np.array([1], dtype=np.int32)
        return mask, class_ids

# Configurations for Inference
class InferenceConfig(Config):
    NAME = "infer_cfg"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2  # Background + 1 disease class
    BACKBONE = "resnet50"

inference_config = InferenceConfig()

# Prepare the test dataset
dataset_test = DiseaseDataset()
dataset_test.load_dataset(DATASET_DIR, "test")
dataset_test.prepare()

# Load the trained Mask R-CNN model in inference mode
model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)
model.load_weights(BEST_MODEL_PATH, by_name=True)

def evaluate_model_and_visualize(model, dataset, config):
    """Evaluate model and visualize predictions."""
    ious = []
    all_true_masks = []
    all_pred_masks = []
    tp, fp, fn, tn = 0, 0, 0, 0

    for image_id in dataset.image_ids:
        # Load the image and true mask
        image = dataset.load_image(image_id)
        true_mask, _ = dataset.load_mask(image_id)
        
        # Run detection
        result = model.detect([image], verbose=0)[0]
        pred_mask = result['masks']
        
        # Ensure the predicted mask matches the image size if necessary
        if pred_mask.shape[:2] != image.shape[:2]:
            pred_mask = np.resize(pred_mask, image.shape[:2] + (pred_mask.shape[-1],))
        
        # Combine all predicted masks (multi-channel) into a single binary mask
        combined_pred_mask = np.sum(pred_mask, axis=-1)
        combined_pred_mask = (combined_pred_mask > 0).astype(np.uint8)
        
        # Calculate IoU for each instance
        if combined_pred_mask.sum() > 0 and true_mask.shape[-1] > 0:
            intersection = np.logical_and(combined_pred_mask, true_mask[:, :, 0])
            union = np.logical_or(combined_pred_mask, true_mask[:, :, 0])
            iou = np.sum(intersection) / np.sum(union)
            ious.append(iou)
            
            # Append masks for confusion matrix calculation
            all_true_masks.append(true_mask[:, :, 0])
            all_pred_masks.append(combined_pred_mask)
            
            # Calculate TP, FP, FN, TN
            tp += np.sum(np.logical_and(combined_pred_mask, true_mask[:, :, 0]))
            fp += np.sum(np.logical_and(combined_pred_mask, np.logical_not(true_mask[:, :, 0])))
            fn += np.sum(np.logical_and(np.logical_not(combined_pred_mask), true_mask[:, :, 0]))
            tn += np.sum(np.logical_and(np.logical_not(combined_pred_mask), np.logical_not(true_mask[:, :, 0])))
        
        # Generate visualizations
        visualize_image_masks(image, true_mask, pred_mask, image_id)
    
    # Compute and display the confusion matrix
    compute_confusion_matrix(all_true_masks, all_pred_masks)
    
    # Calculate and save metrics
    mean_iou = np.mean(ious)
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    save_metrics_to_file(mean_iou, tp, fp, fn, tn, accuracy, precision, recall)

def visualize_image_masks(image, true_mask, pred_mask, image_id):
    """Visualize original image, true mask, predicted mask, and overlay with white overlay."""
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    
    # Ensure the predicted mask matches the image size if necessary
    if pred_mask.shape[:2] != image.shape[:2]:
        pred_mask = np.resize(pred_mask, image.shape[:2] + (pred_mask.shape[-1],))
    
    # Create a combined prediction mask (sum over all mask channels)
    combined_pred_mask = np.sum(pred_mask, axis=-1)
    combined_pred_mask = (combined_pred_mask > 0).astype(np.uint8)  # Make binary
    
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')  # Hide axis with pixel counts
    
    ax[1].imshow(true_mask[:, :, 0], cmap='gray')
    ax[1].set_title("True Mask")
    ax[1].axis('off')  # Hide axis with pixel counts
    
    if combined_pred_mask.sum() > 0:
        ax[2].imshow(combined_pred_mask, cmap='gray')
        ax[2].set_title("Predicted Mask")
        ax[2].axis('off')  # Hide axis with pixel counts
        
        # Create a white overlay for the predicted mask
        overlay = np.copy(image)
        overlay[combined_pred_mask > 0] = [255, 255, 255]  # Apply white mask
        
        # Display the overlay image
        ax[3].imshow(overlay)
        ax[3].set_title("Overlay of Prediction")
        ax[3].axis('off')  # Hide axis with pixel counts
    else:
        ax[2].imshow(np.zeros_like(true_mask[:, :, 0]), cmap='gray')
        ax[2].set_title("Predicted Mask (None)")
        ax[2].axis('off')  # Hide axis with pixel counts
        ax[3].imshow(image)
        ax[3].set_title("Overlay of Prediction (None)")
        ax[3].axis('off')  # Hide axis with pixel counts

    # Save visualization to PNG without pixel counts
    plt.savefig(os.path.join(RESULT_DIR, f"visualization_{image_id}.png"))
    plt.close(fig)

def compute_confusion_matrix(true_masks, pred_masks):
    """Compute and plot confusion matrix for all masks."""
    # Flatten the mask arrays
    true_masks_flat = np.concatenate([mask.flatten() for mask in true_masks])
    pred_masks_flat = np.concatenate([mask.flatten() for mask in pred_masks])
    
    # Compute confusion matrix
    cm = confusion_matrix(true_masks_flat, pred_masks_flat, labels=[0, 1])
    plot_confusion_matrix(cm, classes=["Background", "Plant Disease"])

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """Plot and display the confusion matrix with both pixel counts and percentages."""
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm, cmap=cmap)
    plt.title(title)
    fig.colorbar(cax)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f'{cm[i, j]} ({cm_percent[i, j]:.2f}%)',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Save confusion matrix to file
    plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"))
    plt.show()

def save_metrics_to_file(mean_iou, tp, fp, fn, tn, accuracy, precision, recall):
    """Save evaluation metrics to a text file."""
    metrics_file = os.path.join(RESULT_DIR, "evaluation_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"Mean IoU: {mean_iou:.4f}\n")
        f.write(f"True Positives (TP): {tp}\n")
        f.write(f"False Positives (FP): {fp}\n")
        f.write(f"False Negatives (FN): {fn}\n")
        f.write(f"True Negatives (TN): {tn}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")

# Run the evaluation and visualization
evaluate_model_and_visualize(model, dataset_test, inference_config)