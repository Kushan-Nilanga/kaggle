from torch.utils.data.dataset import Dataset
import pandas as pd
import pydicom
import os
import json
import torch

class ImageDataset(Dataset):
    def __init__(self, csv_file, dicom_dir):
        super(ImageDataset, self).__init__()
        self.csv_file = pd.read_csv(csv_file)
        self.dicom_dir = dicom_dir
        self.image_dic = {}
        for idx, row in self.csv_file.iterrows():
            for root, _, files in os.walk(os.path.join(self.dicom_dir, row.StudyInstanceUID), topdown=True):
                for name in files:
                    self.image_dic[name.split(
                        '.')[0]] = os.path.join(root, name)

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, index):
        csv_row = self.csv_file.iloc[index]
        image = csv_row[0].split("_")[0]
        im_dcm = pydicom.dcmread(self.image_dic[image])
        boxes = None
        try:
            boxes = json.loads(csv_row['boxes'].replace("'", "\""))
            placeholder = torch.zeros((len(boxes), 4), dtype=torch.float32)

            for i in range(len(boxes)):
                placeholder[i, 0] = boxes[i]['x']
                placeholder[i, 1] = boxes[i]['y']
                placeholder[i, 2] = boxes[i]['width']
                placeholder[i, 3] = boxes[i]['height']

            boxes = placeholder

        except:
            pass

        return (torch.tensor(im_dcm.pixel_array.astype(int), dtype=torch.float32), boxes)
