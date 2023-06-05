import os
import numpy as np
from pathlib import Path

import torch
from torchvision.transforms import transforms
from torchvision.io import read_image

class BreastCancerDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, magnitude = "400X", transform = None):
        self.image_paths = [x for x in image_folder.rglob("*.png") if magnitude in x.parent.name]
        self.labels = [x.relative_to(image_folder).parts[0] for x in self.image_paths]
        self.classes = list(set(self.labels))
        self.class_to_label = {x:i for i, x in enumerate(self.classes)}
        self.label_to_class = {v:k for k, v in self.class_to_label.items()}
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = read_image(str(image_path))
        label = self.class_to_label[self.labels[idx]]
        if self.transform:
            image = self.transform(image)
        return image.float(), label