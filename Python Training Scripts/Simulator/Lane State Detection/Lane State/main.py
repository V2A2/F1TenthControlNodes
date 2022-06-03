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
numberOfDataPointsStored = 1
numberOfImageClassifications = 4  # change to 6 for real car
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

class CustomImageDataset(Dataset):
    def __init__(self, filePath, transform=None, target_transform =None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = Path(filePath).read_bytes()
        self.labels = torch.zeros([int(self.data.__len__()/(steeringAngleByteSize*numberOfDataPointsStored)),2], dtype=torch.float)
        self.images = torch.zeros([int(self.data.__len__()/(steeringAngleByteSize*numberOfDataPointsStored)),1, width, height], dtype=torch.float)
        for idx in range(int(self.data.__len__()/(steeringAngleByteSize*numberOfDataPointsStored))):
            unmodifiedCameraImage = cv.imread('/home/ryan/TrainingData/simulator_lane_state/photos/' + str(idx) + "-semanticOutput.png", cv.IMREAD_GRAYSCALE)
            #cameraImage = cv.resize(unmodifiedCameraImage, (width, height))
            cameraTensor = torch.from_numpy(unmodifiedCameraImage)
            inputToNetwork = cameraTensor.unsqueeze(0).to(device).type(torch.float)
            finalPrediction = cameraTensor.mul(1/256)
            # if idx %100 == 0:
            #     plt.figure()
            #     plt.imshow(finalPrediction.mul(40).to("cpu").squeeze())
            self.images[idx] = finalPrediction#semanticSegmentationImage
            inputLaneState = struct.unpack('d', self.data[(steeringAngleByteSize*numberOfDataPointsStored)*idx:steeringAngleByteSize+(steeringAngleByteSize*numberOfDataPointsStored)*idx])[0]
            # if round(inputLaneState) == 1:
            #     self.labels[idx] = torch.tensor([0, 1])
            # if round(inputLaneState) == 2:
            #     self.labels[idx] = torch.tensor([0, 0])
            # if round(inputLaneState) == 3:
            #     self.labels[idx] = torch.tensor([1, 1])
            # if round(inputLaneState) == 4:
            #     self.labels[idx] = torch.tensor([1, 0])

            # if round(inputLaneState) == 1:
            #     self.labels[idx] = torch.tensor([1, 0, 0, 0])
            # if round(inputLaneState) == 2:
            #     self.labels[idx] = torch.tensor([0, 1, 0, 0])
            # if round(inputLaneState) == 3:
            #     self.labels[idx] = torch.tensor([0, 0, 1, 0])
            # if round(inputLaneState) == 4:
            #     self.labels[idx] = torch.tensor([0, 0, 0, 1])

            if int(round(inputLaneState)) == 1 or int(round(inputLaneState)) == 2:
                self.labels[idx] = torch.tensor([0, 1])
            elif int(round(inputLaneState)) == 3 or int(round(inputLaneState)) == 4:
                self.labels[idx] = torch.tensor([1, 0])

            print(f"Loaded image {idx}")

    def __len__(self):
        return int(self.data.__len__() / (steeringAngleByteSize*numberOfDataPointsStored))

    def __getitem__(self, idx):
        input = self.images[idx]
        steeringAngle = self.labels[idx]
        return (input, steeringAngle)

dataset = CustomImageDataset("/home/ryan/TrainingData/simulator_lane_state/laneState.bin")
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
            nn.Linear(in_features=10, out_features=2),
        )
    def forward(self, input):
        output = self.conv_layer(input)
        output = self.linear_layers(output)
        return output
model = Model().to(device)
test = torch.randn(16,1,256,256).to(device)
print(model(test).shape)


# loss_function = nn.MSELoss()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
n_total_steps = len(train_dataset)
num_total_steps=len(train_loader)
num_total_steps_in_test=len(test_loader)
highestScore = 0
for epoch in range(num_epochs):
    averageLoss = 0.0
    for i, (images, angles) in enumerate(train_loader):
        images = images.to(device)
        angles = angles.to(device)
        outputs = model(images)
        #print(outputs)
        loss = loss_function(outputs, angles)
        averageLoss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    averageLoss /= num_total_steps*batch_size
    print(f"Epoch: {epoch} Loss: {averageLoss}")
    with torch.no_grad():
        numberCorrect = 0
        numberOfPredictions = 0
        for i, (testImages, testStates) in enumerate(test_loader):
            testImages = testImages.to(device)
            testStates = testStates.to(device)
            testOutputs = model(testImages)
            for index in range(len(testOutputs)):
                test3 = testOutputs[index]
                test1 = testOutputs[index].argmax(0)
                test2 = testStates[index].argmax(0)
                if testOutputs[index].argmax(0) == testStates[index].argmax(0):
                    numberCorrect += 1
                numberOfPredictions += 1
        accuracy = numberCorrect / numberOfPredictions
        if accuracy > highestScore:
            PATH = './train_network'+str(epoch)+'.pth'
            torch.save(model.state_dict(), PATH)
            highestScore = accuracy

        print(f"Accuracy: {accuracy}")


print('Finished Training')
