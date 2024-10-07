import torch
import torch.nn as nn


class Detection_network(nn.Module):
    def __init__(self, input_channels, n_classes):
        super(Detection_network, self).__init__()
        # Inspired by YOLO model
        # Bloc1 
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, stride=1)      # dim = 53
        self.leaky1 = nn.LeakyReLU()
        self.maxpol1 = nn.MaxPool2d(kernel_size=3, padding=0, stride=2)                                 # din = 26 
        # Bloc2
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.leaky2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.leaky3 = nn.LeakyReLU()
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.leaky4 = nn.LeakyReLU()
        self.maxpol2 = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)                                 # din = 13
        # Bloc3
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.leaky5 = nn.LeakyReLU()
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.leaky6 = nn.LeakyReLU()
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.leaky8 = nn.LeakyReLU()
        self.maxpol3 = nn.MaxPool2d(kernel_size=3, padding=0, stride=2)                                 # din = 5
        # Bloc4 
        self.conv9 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.leaky9 = nn.LeakyReLU()
        self.conv10 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.leaky10 = nn.LeakyReLU()
        self.conv11 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.leaky11 = nn.LeakyReLU()
        # Bloc5 
        self.conv12 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=0, stride=1)
        self.relu = nn.ReLU()
        self.conv13 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky1(x)
        x = self.maxpol1(x)

        x = self.conv2(x)
        x = self.leaky2(x)
        x = self.conv3(x)
        x = self.leaky3(x)
        x = self.conv4(x)
        x = self.leaky4(x)
        x = self.maxpol2(x)

        x = self.conv5(x)
        x = self.leaky5(x)
        x = self.conv6(x)
        x = self.leaky6(x)
        x = self.conv7(x)
        x = self.leaky8(x)
        x = self.maxpol3(x)

        x = self.conv9(x)
        x = self.leaky9(x)
        x = self.conv10(x)
        x = self.leaky10(x)
        x = self.conv11(x)
        x = self.leaky11(x)

        x = self.conv12(x)
        x = self.relu(x)
        x = self.conv13(x) # shape = 16*2*2

        output = x.view(x.size(0) * x.size(1), 3, 5)

        return output
