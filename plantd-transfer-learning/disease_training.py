import os
import numpy as np
import skimage.io
import xml.etree.ElementTree as ET
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import config
# run while working directory contains both the dataset folder + coco weights file.
class PlantDiseaseDataset(utils.Dataset):
    def load_dataset(self, dataset_dir, is_train=True):
        self.add_class("dataset", 1, "disease")
        
        images_dir = os.path.join(dataset_dir, 'images')
        annotations_dir = os.path.join(dataset_dir, 'annots')
        masks_dir = os.path.join(dataset_dir, 'masks')
        
        all_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        all_ids = [os.path.splitext(fname)[0] for fname in all_files]
        all_ids = sorted(all_ids)  # Ensure consistent order

        split_index = int(0.8 * len(all_ids))  # 80%-20% train-val split
        if is_train:
            ids = all_ids[:split_index]
        else:
            ids = all_ids[split_index:]

        for image_id in ids:
            img_path = os.path.join(images_dir, image_id + '.jpg')
            ann_path = os.path.join(annotations_dir, image_id + '.xml')
            mask_path = os.path.join(masks_dir, image_id + '.png')

            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path, mask_path=mask_path)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask_path = info['mask_path']
        annotation_path = info['annotation']

        mask = skimage.io.imread(mask_path, as_gray=True)
        mask = mask > 128  # Convert to boolean mask

        boxes = self.extract_boxes(annotation_path)
        count = len(boxes)
        masks = np.zeros((mask.shape[0], mask.shape[1], count), dtype=np.bool)

        class_ids = []
        for i, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = box
            masks[ymin:ymax, xmin:xmax, i] = mask[ymin:ymax, xmin:xmax]
            class_ids.append(self.class_names.index('disease'))

        return masks, np.array(class_ids, dtype=np.int32)

    def extract_boxes(self, filename):
        tree = ET.parse(filename)
        root = tree.getroot()
        boxes = []
        for member in root.findall('.//bndbox'):
            xmin = int(member.find('xmin').text)
            ymin = int(member.find('ymin').text)
            xmax = int(member.find('xmax').text)
            ymax = int(member.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
        return boxes

class PlantDiseaseConfig(config.Config):
    NAME = "disease_cfg"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + disease
    STEPS_PER_EPOCH = 100

# Setup paths in an OS-agnostic way
dataset_root_path = os.path.join(os.getcwd(), 'data')

train_dataset = PlantDiseaseDataset()
train_dataset.load_dataset(dataset_root_path, is_train=True)
train_dataset.prepare()

validation_dataset = PlantDiseaseDataset()
validation_dataset.load_dataset(dataset_root_path, is_train=False)
validation_dataset.prepare()

config = PlantDiseaseConfig()
model = modellib.MaskRCNN(mode='training', model_dir='./', config=config)
model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
#set to 1 epoch for testing. eventually set to same as deeplabv3+. maybe 25?
model.train(train_dataset, validation_dataset, learning_rate=config.LEARNING_RATE, epochs=1, layers='heads')

model_path = 'plant_disease_mask_rcnn.h5'
model.keras_model.save_weights(model_path)