import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class AnimalDataset(Dataset):
    def __init__(self, root, train=False,transform = None):
        self.train = train
        self.transform = transform
        self.images = []
        self.labels = []
        if self.train == True :
            root = f'{root}/train'
        else :
            root = f'{root}/test'
        self.types =os.listdir(root).sort()
        
        for type in self.types:
            data_folder = f'{root}/{type}'
            
            files = os.listdir(data_folder)
            len_files = len(files)
        
            for splt in range(len_files):
                file = files[splt]
                path = os.path.join(data_folder, file)
                self.images.append(path)
                self.labels.append(self.types.index(type))
                # break
            # break
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        if self.transform :
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label
    def get_type (self):
        return self.types