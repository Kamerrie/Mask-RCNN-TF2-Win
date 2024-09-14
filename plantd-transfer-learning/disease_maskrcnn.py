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
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('WARNING')

DATASET_DIR = os.path.join(os.getcwd(), "data")
MASKRCNN_DIR = os.path.join(os.getcwd(), "maskrcnn")
MODEL_DIR = os.path.join(MASKRCNN_DIR, "models")
LOG_DIR = os.path.join(MASKRCNN_DIR, "logs")
WEIGHTS_DIR = os.path.join(MODEL_DIR, "weights")
BEST_MODEL_PATH = os.path.join(WEIGHTS_DIR, "best_model.h5")
EPOCH_FILE_PATH = os.path.join(MODEL_DIR, "last_epoch.txt")
BEST_IOU_FILE_PATH = os.path.join(MODEL_DIR, "best_best_iou.txt")
WAIT_COUNTER_FILE_PATH = os.path.join(MODEL_DIR, "wait_counter.txt")

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
if not os.path.exists(WEIGHTS_DIR):
    os.makedirs(WEIGHTS_DIR)

total_epochs = 100
total_train_images = len([name for name in os.listdir(os.path.join(DATASET_DIR, "train", "images")) if os.path.isfile(os.path.join(os.path.join(DATASET_DIR, "train", "images"), name)) and name.endswith('.jpg')])
total_val_images = len([name for name in os.listdir(os.path.join(DATASET_DIR, "val", "images")) if os.path.isfile(os.path.join(os.path.join(DATASET_DIR, "val", "images"), name)) and name.endswith('.jpg')])


def read_last_epoch(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return int(f.read().strip())
    return 0


def write_last_epoch(file_path, epoch):
    with open(file_path, 'w') as f:
        f.write(str(epoch))


def read_best_iou(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return float(f.read().strip())
    return 0


def write_best_iou(file_path, mean_iou):
    with open(file_path, 'w') as f:
        f.write(str(mean_iou))


def read_wait_counter(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return int(f.read().strip())
    return 0


def write_wait_counter(file_path, wait):
    with open(file_path, 'w') as f:
        f.write(str(wait))

initial_epoch = read_last_epoch(EPOCH_FILE_PATH)
best_iou = read_best_iou(BEST_IOU_FILE_PATH)
wait_counter = read_wait_counter(WAIT_COUNTER_FILE_PATH)

class DiseaseDataset(mrcnn.utils.Dataset):
    def load_dataset(self, dataset_dir, subset):
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
        info = self.image_info[image_id]
        mask_path = info['mask_path']
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)
        mask = np.expand_dims(mask, axis=-1)
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

dataset_train = DiseaseDataset()
dataset_train.load_dataset(DATASET_DIR, "train")
dataset_train.prepare()

dataset_val = DiseaseDataset()
dataset_val.load_dataset(DATASET_DIR, "val")
dataset_val.prepare()

model = mrcnn.model.MaskRCNN(mode="training", config=training_config, model_dir=MODEL_DIR)

class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_file=os.path.join(LOG_DIR, "maskrcnn.log"), patience=10):
        super(MetricsCallback, self).__init__()
        self.log_file = log_file
        self.start_time = None
        self.best_iou = best_iou
        self.best_checkpoint_path = BEST_MODEL_PATH
        self.patience = patience
        self.wait = wait_counter

        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write(
                    "start_time,epoch,end_time,epoch_duration,loss,val_loss,mean_iou,mean_precision,mean_recall,mean_f1_score\n")

    def on_train_begin(self, logs=None):
        self.start_time = datetime.now()
        formatted_start_time = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Training started at: {formatted_start_time}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        write_last_epoch(EPOCH_FILE_PATH, read_last_epoch(EPOCH_FILE_PATH) + 1)
        print(f'\nEpoch {read_last_epoch(EPOCH_FILE_PATH)} Metrics:')
        end_time = datetime.now()
        epoch_duration = (end_time - self.start_time).total_seconds()
        formatted_end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")

        model_name = f'Disease_mask_rcnn_epoch_{read_last_epoch(EPOCH_FILE_PATH)}.h5'
        model_path = os.path.join(WEIGHTS_DIR, model_name)
        self.model.save_weights(model_path)

        inference_model = mrcnn.model.MaskRCNN(mode='inference',
                                               model_dir=MODEL_DIR,
                                               config=inference_config)
        inference_model.load_weights(model_path, by_name=True)

        val_iou = []
        precisions, recalls, f1_scores = [], [], []

        for image_id in dataset_val.image_ids:
            image = dataset_val.load_image(image_id)
            mask, _ = dataset_val.load_mask(image_id)
            results = inference_model.detect([image], verbose=0)
            pred_mask = results[0]['masks']

            if pred_mask.shape[-1] > 0 and mask.shape[-1] > 0:
                iou_matrix = np.zeros((pred_mask.shape[-1], mask.shape[-1]))
                for i in range(pred_mask.shape[-1]):
                    for j in range(mask.shape[-1]):
                        intersection = np.logical_and(pred_mask[:, :, i], mask[:, :, j])
                        union = np.logical_or(pred_mask[:, :, i], mask[:, :, j])
                        iou = np.sum(intersection) / np.sum(union)
                        iou_matrix[i, j] = iou

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
        loss = logs.get('loss', 'N/A')
        val_loss = logs.get('val_loss', 'N/A')

        if mean_iou_value > self.best_iou:
            print(f"Mean IoU improved from {self.best_iou:.4f} to {mean_iou_value:.4f}. Saving model checkpoint.")
            self.best_iou = mean_iou_value
            self.model.save_weights(self.best_checkpoint_path)
            write_best_iou(BEST_IOU_FILE_PATH, mean_iou_value)
            self.wait = 0
        else:
            self.wait += 1
            print(f"No improvement in IoU for {self.wait} epoch(s).")
        write_wait_counter(WAIT_COUNTER_FILE_PATH, self.wait)

        if self.wait >= self.patience:
            print(f"Stopping training after {self.patience} epochs without improvement in IoU.")
            self.model.stop_training = True

        with open(self.log_file, 'a') as f:
            f.write(f"{self.start_time.strftime('%Y-%m-%d %H:%M:%S')},{read_last_epoch(EPOCH_FILE_PATH)},{formatted_end_time},{epoch_duration:.2f},{loss:.4f},{val_loss:.4f},{mean_iou_value:.4f},{mean_precision:.4f},{mean_recall:.4f},{mean_f1_score:.4f}\n")

        self.start_time = datetime.now()

        for metric_name, metric_value in logs.items():
            print(f'{metric_name}: {metric_value:.4f}')

metrics_callback = MetricsCallback(patience=10)

if os.path.exists(BEST_MODEL_PATH):
    print(f"Loading best model weights from: {BEST_MODEL_PATH}")
    model.load_weights(BEST_MODEL_PATH, by_name=True)
else:
    print("No best model found, starting from scratch.")

for epoch in range(initial_epoch, total_epochs):
    print(f"epoch: {epoch}, init epoch: {initial_epoch}, total epoch: {total_epochs}, wait counter: {wait_counter}")
    if(wait_counter < 10):
        model.train(dataset_train, dataset_val,
                learning_rate=training_config.LEARNING_RATE,
                epochs=total_epochs-initial_epoch,
                layers='heads',
                custom_callbacks=[metrics_callback])
