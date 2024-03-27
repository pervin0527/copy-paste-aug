import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

from tqdm import tqdm
from torch.utils.data import Dataset

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
  ]

VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
  ]


class VOCDataset(Dataset):
    def __init__(self, data_dir, year, set_type, transform):
        self.data_dir = data_dir
        self.year = year
        self.set_type = set_type
        self.transform = transform

        self.set_txt = f"{self.data_dir}/VOC{year}/ImageSets/Main/{set_type}.txt"
        self.image_dir = f"{self.data_dir}/VOC{year}/JPEGImages"
        self.annot_dir = f"{self.data_dir}/VOC{year}/Annotations"
        self.mask_dir = f"{self.data_dir}/VOC{year}/SegmentationClass"

        with open(self.set_txt, "r") as file:
            total_files = file.read().splitlines()
            self.data_files = self.check_files(total_files)
            print(len(self.data_files))

    def check_files(self, files):
        readable_files = []
        for file_name in tqdm(files):
            if os.path.exists(f"{self.image_dir}/{file_name}.jpg") and os.path.exists(f"{self.annot_dir}/{file_name}.xml") and os.path.exists(f"{self.mask_dir}/{file_name}.png"):
                readable_files.append(file_name)

        return readable_files

    def extract_bounding_boxes(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        bounding_boxes = []    
        for obj in root.iter('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            bounding_boxes.append((xmin, ymin, xmax, ymax))
        
        return bounding_boxes
    
    def convert_to_segmentation_mask(self, mask, make_binary=True):
        """
        mask의 각 픽셀 값이 label의 RGB 값과 같은지를 검사.
        마지막 축(axis=-1)을 기준으로, 각 픽셀의 값이 label과 정확히 일치할 때만 True.
        """
        height, width = mask.shape[:2]
        if make_binary:
            segmentation_mask = np.zeros((height, width), dtype=np.uint8)
            for label_index, label_color in enumerate(VOC_COLORMAP):
                segmentation_mask[:, :, label_index] = np.all(mask == label_color, axis=-1).astype(float)
        else:
            segmentation_mask = np.zeros((height, width, len(VOC_CLASSES)), dtype=np.float32)
            for label_index, label_color in enumerate(self.args.VOC_COLORMAP):
                match = np.all(mask == np.array(label_color), axis=-1)
                segmentation_mask[match] = label_index

        return segmentation_mask
    
    def load_example(self, idx):
        file = self.data_files[idx]
        image = cv2.imread(f"{self.image_dir}/{file}.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = self.extract_bounding_boxes(f"{self.annot_dir}/{file}.jpg")

        mask = cv2.imread(f"{self.mask_dir}/{file}.png")
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = self.convert_to_segmentation_mask(mask, make_binary=True)

        output = {"image" : image, "mask" : mask, "bboxes" : bboxes}

        return self.transform(**output)


if __name__ == "__main__":
    data_dir = "/home/pervinco/Datasets/PASCAL_VOC/VOCDevkit"
    dataset = VOCDataset(data_dir=data_dir, year=2012, set_type="trainval")