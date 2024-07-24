import os
import pandas as pd
import csv
import os
import tarfile
from urllib.parse import urlparse
from urllib.request import urlretrieve
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class Indian(Dataset):
    def __init__(self, root, phase, transform=None):
        self.root = os.path.abspath(root)
        self.datsetPath = self.root
        self.path_images = os.path.join(self.datsetPath, 'IndianFoodImgs')
        self.phase = phase
        self.transform = transform
        self.num_classes = 57
        # define path of csv file

        # define filename of csv file
        file_csv = os.path.join(self.datsetPath, phase + '.csv')
        ingredients_path = os.path.join(self.datsetPath, 'ingredients.csv')
        self.classes = pd.read_csv(ingredients_path).values.tolist()
        self.images = pd.read_csv(file_csv).values.tolist()
        print('[dataset] Indian classification phase={} number of classes={}  number of images={}'.format(phase, len(self.classes), len(self.images)))

    def __getitem__(self, index):
        filename, dish, ingredients, intLabels = self.images[index]
        img = Image.open(os.path.join(self.path_images, filename)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        labels = sorted([int(x) for x in intLabels.split("|")])
        target = torch.zeros(self.num_classes,  dtype=torch.float32) - 1
        # print("Target shape :",target.shape, " labels :",labels)
        target[labels] = 1
        places1 = []
        for x in target:
            if(x == 1):
                places1.append(x)
        
        # print("Places 1 :",len(places1))
        ## torch.from_numpy((np.asarray(target.split("|"))).astype(np.float32))
        data = {'image':img, 'name': filename, 'target': target}
        return data
        # image = {'image': img, 'name': filename}
        # return image, target
        # return (img, filename), target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)
