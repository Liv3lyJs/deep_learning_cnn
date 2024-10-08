import torch
import torch.nn as nn

class Detection_network(nn.Module):
    def __init__(self):
        super(Detection_network, self).__init__()

        # Inspired AlexNet network
        # Feature extraction
        # Bloc1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=11, stride=2, padding=4) # Output 26x26
        self.relu1 = nn.ReLU()
        self.maxpol1 = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)                             # Output 13x13
        # Bloc3
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        # Bloc5
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        self.maxpol5 = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)                             # Output 6x6

        # Classification
        # Bloc 6
        self.fn6 = nn.Flatten()
        self.fl6 = nn.Linear(32*6*6, 256)
        self.relu6 = nn.ReLU()

        # Bloc 7
        self.fl7 = nn.Linear(256, 256)
        self.relu7 = nn.ReLU()

        # Bloc 8
        self.fl8 = nn.Linear(256, 3*(4+3))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpol1(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpol5(x)

        x = self.fn6(x)
        x = self.fl6(x)
        x = self.relu6(x)

        x = self.fl7(x)
        x = self.relu7(x)

        output = self.fl8(x)
        output = output.view(output.size(0), 3, 4 + 3)  # Shape: (batch_size, 3, 7)
        output[:, :, :4] = torch.sigmoid(output[:, :, :4])

        return output
