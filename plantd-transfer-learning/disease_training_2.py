import os
import xml.etree
from numpy import zeros, asarray

import mrcnn.utils
import mrcnn.config
import mrcnn.model

class DiseaseDataset(mrcnn.utils.Dataset):

    def load_dataset(self, dataset_dir, is_train=True):
        self.add_class("dataset", 1, "disease")
        
        images_dir = os.path.join(dataset_dir, 'images')
        annotations_dir = os.path.join(dataset_dir, 'annots')
        
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

            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    # Loads the binary masks for an image.
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation']
        boxes, w, h = self.extract_boxes(path)
        masks = zeros([h, w, len(boxes)], dtype='uint8')

        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('disease'))
        return masks, asarray(class_ids, dtype='int32')

    # A helper method to extract the bounding boxes from the annotation file
    def extract_boxes(self, filename):
        tree = xml.etree.ElementTree.parse(filename)

        root = tree.getroot()

        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)

        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

class DiseaseConfig(mrcnn.config.Config):
    NAME = "disease_cfg"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    NUM_CLASSES = 2

    STEPS_PER_EPOCH = 588


dataset_root_path = os.path.join(os.getcwd(), 'data')

# Train
train_dataset = DiseaseDataset()
train_dataset.load_dataset(dataset_dir=dataset_root_path, is_train=True)
train_dataset.prepare()

# Validation
validation_dataset = DiseaseDataset()
validation_dataset.load_dataset(dataset_dir=dataset_root_path, is_train=False)
validation_dataset.prepare()

# Model Configuration
disease_config = DiseaseConfig()

# Build the Mask R-CNN Model Architecture
model = mrcnn.model.MaskRCNN(mode='training', 
                             model_dir='./', 
                             config=disease_config)

model.load_weights(filepath='mask_rcnn_coco.h5', 
                   by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

model.train(train_dataset=train_dataset, 
            val_dataset=validation_dataset, 
            learning_rate=disease_config.LEARNING_RATE, 
            epochs=1, 
            layers='heads')

model_path = 'Disease_mask_rcnn_trained.h5'
model.keras_model.save_weights(model_path)
