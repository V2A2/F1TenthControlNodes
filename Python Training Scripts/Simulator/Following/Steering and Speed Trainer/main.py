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
import cv2 as cv
start = time.time()

width = 256
height = 256
pixelByteSize = 1
steeringAngleByteSize = 8


batch_size = 16
num_epochs = 500
#learning_rate = 0.001
learning_rate = 0.001
numberOfDataPointsStored = 2
numberOfImageClassifications = 4  # change to 6 for real car
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

class CustomImageDataset(Dataset):
    def __init__(self, filePath, transform=None, target_transform =None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = Path(filePath).read_bytes()
        self.labels = torch.zeros([int(self.data.__len__()/(steeringAngleByteSize*numberOfDataPointsStored))], dtype=torch.float)
        self.images = torch.zeros([int(self.data.__len__()/(steeringAngleByteSize*numberOfDataPointsStored)),1, width, height], dtype=torch.float)
        for idx in range(int(self.data.__len__()/(steeringAngleByteSize*numberOfDataPointsStored))):
            unmodifiedCameraImage = cv.imread('/home/ryan/TrainingData/simulator_following_data/photos/' + str(idx) + "-semanticOutput.png", cv.IMREAD_GRAYSCALE)
            #cameraImage = cv.resize(unmodifiedCameraImage, (width, height))
            cameraTensor = torch.from_numpy(unmodifiedCameraImage)
            inputToNetwork = cameraTensor.unsqueeze(0).to(device).type(torch.float)
            finalPrediction = cameraTensor.mul(1/256)
            if idx %100 == 0:
                plt.figure()
                plt.imshow(finalPrediction.mul(40).to("cpu").squeeze())
            self.images[idx] = finalPrediction#semanticSegmentationImage
            self.labels[idx] = struct.unpack('d', self.data[(steeringAngleByteSize*numberOfDataPointsStored)*idx:steeringAngleByteSize+(steeringAngleByteSize*numberOfDataPointsStored)*idx])[0]
            print(f"Loaded image {idx}")

    def __len__(self):
        return int(self.data.__len__() / (steeringAngleByteSize*numberOfDataPointsStored))

    def __getitem__(self, idx):
        input = self.images[idx]
        steeringAngle = self.labels[idx]
        return (input, steeringAngle)

dataset = CustomImageDataset("/home/ryan/TrainingData/simulator_following_data/followingData.bin")
'''

batch_size = 8
validation_split = .2
shuffle_dataset = True
random_seed= 500

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)
#
# test_dataloader = CustomImageDataset("test.bin", batch_size=64, shuffle=True)
#
'''
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
print(f"Feature number 1: {train_dataset[0][0].size()}")
print(f"Feature number 1: {train_dataset[1][0]}")
print(f"Label number 1: {train_dataset[1][0]}")


# Define model
# Define model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.AvgPool2d(2),
            # 128 128
            nn.Conv2d(1,24, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.AvgPool2d(2),
            # 32 32
            nn.Conv2d(24, 36, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(36, 36, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            # 16 16
            nn.Conv2d(36, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            # 8 8
            nn.Conv2d(48, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            # 4 4
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=64 * 4 * 4, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=1)
        )
    def forward(self, input):
        output = self.conv_layer(input)
        output = self.linear_layers(output)
        return output
model = Model().to(device)
test = torch.randn(16,1,256,256).to(device)
print(model(test).shape)


loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
n_total_steps = len(train_dataset)
num_total_steps=len(train_loader)
num_total_steps_in_test=len(test_loader)
maxWideAccuracy = 0
maxShortAccuracy = 0
for epoch in range(num_epochs):
    averageLoss = 0.0
    numInRange = 0
    numInShortRange = 0;
    for i, (images, angles) in enumerate(train_loader):
        images = images.to(device)
        angles = angles.to(device)
        outputs = model(images)
        #print(outputs)
        loss = loss_function(outputs,angles.unsqueeze(1))
        averageLoss +=loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss / batch_size< 0.01:
            numInRange = numInRange+1
        if loss /batch_size< 0.0025:
            numInShortRange = numInShortRange+1
    averageLoss/=num_total_steps*batch_size
    print(f"Epoch: {epoch} Loss: {averageLoss} Num in range: {numInRange/(num_total_steps*batch_size)}")
    print(f"Epoch: {epoch} Loss: {averageLoss} Num in range short: {numInShortRange / (num_total_steps*batch_size)}")
    with torch.no_grad():
        totalInRange = 0
        totalInShortRange = 0;
        for i, (testImages, testAngles) in enumerate(test_loader):
            testImages = testImages.to(device)
            testAngles = testAngles.to(device)
            testOutputs = model(testImages)
            lossValues = abs(testOutputs.squeeze(1)-testAngles)
            correct = lossValues[lossValues < 0.1];
            correctSmall = lossValues[lossValues < 0.05];
            testLoss = loss_function(testOutputs, testAngles.unsqueeze(1))
            totalInRange = totalInRange + len(correct)
            totalInShortRange = totalInShortRange + len(correctSmall)
        accuracyWide = totalInRange / (num_total_steps_in_test * batch_size)
        accuracyShort = totalInShortRange / (num_total_steps_in_test * batch_size)
        if(maxWideAccuracy <0.8):
            if(accuracyWide > maxWideAccuracy):
                PATH = './train_network'+str(epoch)+'.pth'
                torch.save(model.state_dict(), PATH)
                maxWideAccuracy = accuracyWide
                maxShortAccuracy = accuracyShort
        else:
            if (accuracyShort > maxShortAccuracy):
                PATH = './train_network' + str(epoch) + '.pth'
                torch.save(model.state_dict(), PATH)
                maxWideAccuracy = accuracyWide
                maxShortAccuracy = accuracyShort
        print(f"Accuracy: {accuracyWide}")
        print(f"Accuracy Short: {accuracyShort}")


print('Finished Training')
