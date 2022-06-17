import torch
from torch.utils.data import Dataset
import cv2 as cv
import os

import config


class CustomImageDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.number_of_images = len([name for name in os.listdir('/home/ryan/TrainingData/real_car_semantic_segmentation/masks_machine')])
        self.labels = torch.zeros([self.number_of_images, config.model_image_width, config.model_image_height], dtype=torch.long)
        self.images = torch.zeros([self.number_of_images, 3, config.model_image_width, config.model_image_height])
        for idx in range(self.number_of_images):
            unmodifiedCameraImage = cv.imread('/home/ryan/TrainingData/real_car_semantic_segmentation/img/'+str(idx+1)+".jpg", cv.IMREAD_COLOR)
            cameraImage = cv.resize(unmodifiedCameraImage, (config.model_image_width, config.model_image_height))
            cameraTensor = torch.from_numpy(cameraImage)
            cameraTensor = cameraTensor.permute(2,0,1)
            self.images[idx] = cameraTensor
            unmodifiedlabelImage = cv.cvtColor(cv.imread('/home/ryan/TrainingData/real_car_semantic_segmentation/masks_machine/' + str(idx+1)+".png", cv.IMREAD_COLOR),cv.COLOR_BGR2GRAY)
            labelImage = cv.resize(unmodifiedlabelImage, (config.model_image_width, config.model_image_height))
            labelTensor = torch.from_numpy(labelImage)
            self.labels[idx] = labelTensor
            print(f"Loaded image {idx+1} / {self.number_of_images}")

    def __len__(self):
        return self.number_of_images

    def __getitem__(self, idx):
        input = self.images[idx]
        output = self.labels[idx]
        return (input, output)