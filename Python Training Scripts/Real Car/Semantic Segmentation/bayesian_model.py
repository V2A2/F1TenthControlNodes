import torch
from torch import nn
import torch.nn.functional as F
import config
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5)
    )
class BayesianModel(nn.Module):
    def __init__(self):
        super(BayesianModel, self).__init__()
        self.average_pool = nn.AvgPool2d(2)
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.encoder_conv1 = double_conv(3,16)
        self.encoder_conv2 = double_conv(16,32)
        self.encoder_conv3 = double_conv(32,64)
        self.encoder_conv4 = double_conv(64,128)
        self.decoder_conv1 = double_conv(64 + 128, 64)
        self.decoder_conv2 = double_conv(32 + 64, 32)
        self.decoder_conv3 = double_conv(16 + 32, 16)
        self.end_image_conv = nn.Conv2d(16, config.number_of_classifications, 1)

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

    def get_distribution(self,image):
        forward_passes = torch.zeros([config.distribution_forward_passes, config.number_of_classifications, config.model_image_width, config.model_image_height])
        for index in range(len(forward_passes)):
            forward_passes[index] = F.softmax(self.forward(image).squeeze(0), dim=0)

        expected_value = torch.sum(forward_passes, dim=0).mul(1 / config.distribution_forward_passes)
        squared_expected_value = torch.square(expected_value)

        squared_passes = torch.square(forward_passes)
        average_of_squares = torch.sum(squared_passes,dim=0).mul(1/config.distribution_forward_passes)

        variance = torch.sub(average_of_squares,squared_expected_value)
        return expected_value, variance


