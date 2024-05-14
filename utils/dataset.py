import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset


WIDER_CLASSES = {'__background__': 0, 'face': 1}


class AnnotationProcessor:

    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind = WIDER_CLASSES if class_to_ind is None else class_to_ind
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        bboxes = []

        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            bndbox = []
            for point in ['xmin', 'ymin', 'xmax', 'ymax']:
                bndbox.append(int(bbox.find(point).text))

            cls_idx = self.class_to_ind[name]
            bndbox.append(cls_idx)  # [xmin, ymin, xmax, ymax, label_idx]

            bboxes.append(bndbox)

        bboxes_array = np.array(bboxes, dtype=np.float32)

        return bboxes_array


class VOCDetection(Dataset):

    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = AnnotationProcessor()

        self.image_paths = []
        self.label_paths = []

        with open(os.path.join(self.root, 'img_list.txt'), 'r') as file:
            for line in file.readlines():
                file_path, annotation_path = line.split()
                self.image_paths.append(os.path.join(self.root, "images", file_path))
                self.label_paths.append(os.path.join(self.root, "annotations", annotation_path))

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        # Load and parse the annotation
        annotation = ET.parse(label_path).getroot()

        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # Get image dimensions
        height, width, _ = image.shape

        # Transform the target if a target transform function is provided
        if self.target_transform is not None:
            annotation = self.target_transform(annotation)

        # Apply preprocessing if a preprocessing function is provided
        if self.transform is not None:
            image, annotation = self.transform(image, annotation)

        # Convert image to a PyTorch tensor
        image = torch.from_numpy(image)

        return image, annotation

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def collate_fn(batch):
        images = []
        targets = []
        # Iterate over each data sample in the batch
        for image, target in batch:
            images.append(image)
            targets.append(torch.from_numpy(target).float())

        return torch.stack(images, 0), targets
