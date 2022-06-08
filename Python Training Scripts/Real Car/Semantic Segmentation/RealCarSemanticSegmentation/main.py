import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from pathlib import Path
import struct
import os
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from typing import Optional,Sequence
import cv2 as cv
import pylab
start = time.time()
#image details
width = 256
height = 256
pixelByteSize = 1
colorChannelsOfInput = 3;
sizeOf1ChannelImage = width * height * pixelByteSize
sizeOfInputImage = colorChannelsOfInput  * sizeOf1ChannelImage
sizeOfOutputImage = 1 * sizeOf1ChannelImage
sizeOfIndex = sizeOfInputImage + sizeOfOutputImage
numberOfImageClassifications = 4
batch_size = 16#was 64
num_epochs = 200
learning_rate = 0.001;
#learning_rate = 0.10
numberOfTrainingImages = 298
device = "cuda" if torch.cuda.is_available() else "cpu"
#device = 'cpu'
print("Using {} device".format(device))


class CustomImageDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.labels = torch.zeros([numberOfTrainingImages, width, height], dtype=torch.long)
        self.images = torch.zeros([numberOfTrainingImages, colorChannelsOfInput, width, height])
        for idx in range(numberOfTrainingImages):
            unmodifiedCameraImage = cv.imread('/home/ryan/TrainingDataForRealCar/img/'+str(idx+1)+".jpg", cv.IMREAD_COLOR)
            cameraImage = cv.resize(unmodifiedCameraImage, (width, height))
            cameraTensor = torch.from_numpy(cameraImage)
            cameraTensor = cameraTensor.permute(2,0,1)
            self.images[idx] = cameraTensor
            unmodifiedlabelImage = cv.cvtColor(cv.imread('/home/ryan/TrainingDataForRealCar/masks_machine/' + str(idx+1)+".png", cv.IMREAD_COLOR),cv.COLOR_BGR2GRAY)# I messed up labeling in folder so idx +1 matches images
            labelImage = cv.resize(unmodifiedlabelImage, (width, height))
            labelTensor = torch.from_numpy(labelImage)
            self.labels[idx] = labelTensor
            print(f"Loaded image {idx+1} / {numberOfTrainingImages}")

    def __len__(self):
        return numberOfTrainingImages

    def __getitem__(self, idx):
        input = self.images[idx]
        output = self.labels[idx]
        return (input, output)

dataset = CustomImageDataset()

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

'''
print(f"Feature number 1: {train_dataset[0][0].size()}")
print(f"Feature number 1: {train_dataset[1][0]}")
print(f"Label number 1: {train_dataset[1][0]}")
'''
print(f"Feature number 1: {train_dataset[2][0].size()}")

print(f"Feature number 1: {train_dataset[0][1].size()}")
#repeated feature in unet design
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )
# Define model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.average_pool = nn.AvgPool2d(2)
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.encoder_conv1 = double_conv(3,32)
        # self.encoder_conv2 = double_conv(32,64)
        # self.encoder_conv3 = double_conv(64,128)
        # self.encoder_conv4 = double_conv(128,256)
        # self.decoder_conv1 = double_conv(128 + 256, 128)
        # self.decoder_conv2 = double_conv(64 + 128, 64)
        # self.decoder_conv3 = double_conv(32 + 64, 32)
        # self.end_image_conv = nn.Conv2d(32, numberOfImageClassifications, 1)
        self.encoder_conv1 = double_conv(3,16)
        self.encoder_conv2 = double_conv(16,32)
        self.encoder_conv3 = double_conv(32,64)
        self.encoder_conv4 = double_conv(64,128)
        self.decoder_conv1 = double_conv(64 + 128, 64)
        self.decoder_conv2 = double_conv(32 + 64, 32)
        self.decoder_conv3 = double_conv(16 + 32, 16)
        self.end_image_conv = nn.Conv2d(16, numberOfImageClassifications, 1)

    def forward(self, x):
        conv1 = self.encoder_conv1(x)
        x = self.average_pool(conv1)
        conv2 = self.encoder_conv2(x)
        x = self.average_pool(conv2)
        conv3 = self.encoder_conv3(x)
        x = self.average_pool(conv3)
        x = self.encoder_conv4(x)
        x = self.up_sample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.decoder_conv1(x)
        x = self.up_sample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.decoder_conv2(x)
        x = self.up_sample(x)
        x = torch.cat([x,conv1],dim=1)
        x = self.decoder_conv3(x)
        x = self.end_image_conv(x)
        return x
model = Model().to(device)
#model.load_state_dict(torch.load("pretrainedModel.pth"))
class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return 0.
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
loss_function = FocalLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
n_total_steps = len(train_dataset)
num_total_steps=len(train_loader)
num_total_steps_in_test=len(test_loader)
highestAccuracy = 0.0;
for epoch in range(num_epochs):
    numInRange = 0
    numInShortRange = 0;
    totalLoss = 0
    failedEpochs = 0;
    for i, (images, maskImages) in enumerate(train_loader):
        #plt.imshow(images[0].permute(1, 2, 0))
        images = images.to(device)
        maskImages = maskImages.to(device)
        outputs = model(images)
        #print(outputs)
        #loss = cross_entropy2d(outputs, maskImages)
        loss = loss_function(outputs, maskImages)
        totalLoss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    totalLoss /= num_total_steps
    print(f"Epoch: {epoch} Loss: {totalLoss} ")
    if(epoch%1 == 0):
        shouldRecord = 0
        with torch.no_grad():
            totalCorrect = 0
            numCounted = 0

            for j, (test_images, test_masks) in enumerate(test_loader):
                test_images = test_images.to(device)
                test_masks = test_masks.to(device)
                test_output = model(test_images)
                test_output = test_output.argmax(1)
                if j>=0 and epoch >10 and epoch %10 == 0:
                     f = plt.figure()
                     time.sleep(0.25)
                     f.add_subplot(2, 2, 1)
                     test_display = test_output[0].mul(40).to("cpu").squeeze()
                     plt.imshow(test_display)
                     f.add_subplot(2, 2, 2)
                     test_correct = test_masks[0].mul(40).to("cpu").squeeze()
                     plt.imshow(test_correct)
                     f.add_subplot(2, 2, 3)
                     test_display2 = test_output[3].mul(40).to("cpu").squeeze()
                     plt.imshow(test_display2)
                     f.add_subplot(2, 2, 4)
                     test_correct2 = test_masks[3].mul(40).to("cpu").squeeze()
                     plt.imshow(test_correct2)
                     pylab.show()
                     time.sleep(0.25)

                numCorrect = 0
                for x in range(width):
                    for y in range(height):
                        if test_output[0][x][y] == test_masks[0][x][y]:
                            numCorrect += 1
                totalCorrect+=numCorrect
                numCounted+=1
            accuracy = totalCorrect/(width*height * numCounted)
            print(f"Accuracy: {accuracy}")
            if((accuracy>=highestAccuracy or accuracy>=.995) and epoch >20):
                highestAccuracy = accuracy
                shouldRecord = 1
        if shouldRecord:
            PATH = './train_network' + str(epoch) + '.pth'
            torch.save(model.state_dict(), PATH)
            print("Saved Model")

print('Finished Training')

PATH = './train_network.pth'
torch.save(model.state_dict(), PATH)


