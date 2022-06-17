import config
from dataset import CustomImageDataset
from loss import FocalLoss
from bayesian_model import BayesianModel
import torch
import matplotlib.pyplot as plt
import time
import pylab

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = CustomImageDataset()
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

model = BayesianModel().to(device)

loss_function = FocalLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate)
num_total_steps=len(train_loader)
num_total_steps_in_test=len(test_loader)
highestAccuracy = 0.0

for epoch in range(config.num_epochs):
    totalLoss = 0
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

    if epoch % 1 == 0:
        shouldRecord = False
        with torch.no_grad():

            totalCorrect = 0
            numCounted = 0

            for j, (test_images, test_masks) in enumerate(test_loader):
                test_images = test_images.to(device)
                test_masks = test_masks.to(device)
                test_output = model(test_images)
                test_output = test_output.argmax(1)
                if j>=0 and epoch > 10 and epoch % 80 == 0:
                    sample_1_expected_value, sample_1_variance = model.get_distribution(test_images[0].unsqueeze(0))
                    sample_2_expected_value, sample_2_variance = model.get_distribution(test_images[1].unsqueeze(0))
                    prediction_1 = sample_1_expected_value.argmax(0)
                    prediction_2 = sample_2_expected_value.argmax(0)
                    confidence_tensor_1 = torch.zeros(prediction_1.shape)
                    for row in range(confidence_tensor_1.shape[0]):
                        for col in range(confidence_tensor_1.shape[1]):
                            confidence_tensor_1[row][col] = sample_1_variance[prediction_1[row][col]][row][col]
                    confidence_tensor_2 = torch.zeros(prediction_2.shape)
                    for row in range(confidence_tensor_2.shape[0]):
                        for col in range(confidence_tensor_2.shape[1]):
                            confidence_tensor_2[row][col] = sample_2_variance[prediction_2[row][col]][row][col]

                    f = plt.figure()
                    # Image 1
                    time.sleep(0.25)
                    f.add_subplot(2, 3, 1)
                    plt.imshow(prediction_1)
                    f.add_subplot(2, 3, 2)
                    plt.imshow(confidence_tensor_1)
                    f.add_subplot(2, 3, 3)
                    plt.imshow(test_images[0].permute(1,2,0).mul(1/256).to("cpu"))

                    # Image 2
                    f.add_subplot(2, 3, 4)
                    plt.imshow(prediction_2)
                    f.add_subplot(2, 3, 5)
                    plt.imshow(confidence_tensor_2)
                    f.add_subplot(2, 3, 6)
                    plt.imshow(test_images[1].permute(1,2,0).mul(1/256).to("cpu"))

                    pylab.show()
                    time.sleep(0.25)

                numCorrect = 0
                for x in range(config.model_image_width):
                    for y in range(config.model_image_height):
                        if test_output[0][x][y] == test_masks[0][x][y]:
                            numCorrect += 1

                totalCorrect += numCorrect
                numCounted += 1
            accuracy = totalCorrect/(config.model_image_width * config.model_image_height * numCounted)
            print(f"Accuracy: {accuracy}")
            if(accuracy >= highestAccuracy or accuracy >= .995) and epoch >80:
                highestAccuracy = accuracy
                shouldRecord = True
            if shouldRecord:
                PATH = './saved_models/train_network_cpp' + str(epoch) + '.pt'
                torch.save(model.state_dict(), PATH)
                PATH = './saved_models/train_network_python' + str(epoch) + '.pth'
                torch.save(model.state_dict(), PATH)
                print("Saved Model")

